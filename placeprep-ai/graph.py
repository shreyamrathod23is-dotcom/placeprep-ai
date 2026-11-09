"""LangGraph pipelines + OpenRouter LLM client for PlacePrep AI.

Three compiled graphs:
  1. question graph : extract_skills -> generate_questions -> review_questions
                      (conditional edge loops back to the generator at most once)
  2. eval graph     : evaluate_answer -> (score < 5 ? generate_followup) -> END
  3. plan graph     : recommend
"""
import hashlib
import json
import os
import re
import threading
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)
from pydantic import BaseModel, Field, ValidationError, field_validator

import prompts

load_dotenv()

_DEFAULT_MODEL = "meta-llama/llama-3.3-70b-instruct:free"
# OPENROUTER_MODEL may be a comma-separated list; later entries are fallbacks.
MODELS = [m.strip() for m in os.getenv("OPENROUTER_MODEL", _DEFAULT_MODEL).split(",")
          if m.strip()] or [_DEFAULT_MODEL]
MODEL = MODELS[0]
BASE_URL = "https://openrouter.ai/api/v1"
MAX_CHARS = 6000          # truncate long resumes / JDs to keep prompts small
RELEVANCE_THRESHOLD = 6   # reviewer scores below this are "weak"
MAX_REVISIONS = 1         # reviewer may send questions back only once
FOLLOWUP_THRESHOLD = 5    # answers scoring below this trigger a follow-up question


class LLMError(Exception):
    """Friendly, user-displayable error raised for any LLM/API problem.

    fatal=True means retrying or switching model cannot help (bad key, no network).
    """

    def __init__(self, message: str, fatal: bool = False):
        super().__init__(message)
        self.fatal = fatal


# --------------------------------------------------------------------------- #
# Pydantic models = JSON validation for every LLM response
# --------------------------------------------------------------------------- #
def _coerce_score(v):
    """Accept ints, floats or numeric strings; clamp to 0-10."""
    try:
        return max(0, min(10, int(round(float(v)))))
    except (TypeError, ValueError):
        raise ValueError("score must be a number between 0 and 10")


class SkillProfile(BaseModel):
    skills: list[str]
    jd_requirements: list[str] = []
    gaps: list[str] = []
    experience_level: str = "fresher"


class Question(BaseModel):
    question: str
    topic: str
    difficulty: str = "medium"


class QuestionSet(BaseModel):
    questions: list[Question] = Field(min_length=1)


class QuestionReview(BaseModel):
    index: int
    score: int
    reason: str = ""

    @field_validator("score", mode="before")
    @classmethod
    def _clamp_score(cls, v):
        return _coerce_score(v)


class ReviewSet(BaseModel):
    reviews: list[QuestionReview]


class Evaluation(BaseModel):
    score: int
    strengths: list[str] = []
    missed_concepts: list[str] = []
    model_answer: str

    @field_validator("score", mode="before")
    @classmethod
    def _clamp_score(cls, v):
        return _coerce_score(v)


class FollowUp(BaseModel):
    question: str


class ResumeRewrite(BaseModel):
    original: str
    improved: str
    why: str = ""


class ResumeTips(BaseModel):
    missing_keywords: list[str] = []
    rewrites: list[ResumeRewrite] = Field(min_length=1)
    summary: str = ""


class WeakTopic(BaseModel):
    topic: str
    why: str = ""
    resources: list[str] = []


class DayPlan(BaseModel):
    day: int
    focus: str
    tasks: list[str] = []


class ImprovementPlan(BaseModel):
    weak_topics: list[WeakTopic] = Field(min_length=1)
    study_plan: list[DayPlan] = Field(min_length=1)


# --------------------------------------------------------------------------- #
# Response cache: identical prompts return instantly (saves rate-limited free calls)
# --------------------------------------------------------------------------- #
CACHE_FILE = Path(__file__).parent / ".cache" / "llm_cache.json"
cache_enabled = os.getenv("LLM_CACHE", "1") != "0"   # the UI sidebar can toggle this
cache_stats = {"hits": 0, "misses": 0}
_cache: dict | None = None
_cache_lock = threading.Lock()                       # evaluations run in parallel threads


def _cache_key(system: str, user: str, model_cls: type[BaseModel], temperature: float) -> str:
    """Key covers prompts, schema, temperature and the model list, so editing any of them
    naturally invalidates old entries."""
    blob = json.dumps([MODELS, system, user, model_cls.__name__, temperature])
    return hashlib.sha256(blob.encode()).hexdigest()


def _cache_get(key: str):
    global _cache
    with _cache_lock:
        if _cache is None:
            try:
                _cache = json.loads(CACHE_FILE.read_text())
            except (OSError, ValueError):
                _cache = {}
        return _cache.get(key)


def _cache_put(key: str, value: dict) -> None:
    with _cache_lock:
        _cache[key] = value
        try:  # best-effort persistence; the cache is never required for correctness
            CACHE_FILE.parent.mkdir(exist_ok=True)
            tmp = CACHE_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps(_cache))
            tmp.replace(CACHE_FILE)
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# LLM client helpers
# --------------------------------------------------------------------------- #
def _client() -> OpenAI:
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise LLMError("OPENROUTER_API_KEY is missing. Copy .env.example to .env and add your key.",
                       fatal=True)
    return OpenAI(api_key=key, base_url=BASE_URL, default_headers={"X-Title": "PlacePrep AI"})


def _chat(messages: list[dict], temperature: float, model: str) -> str:
    """One chat-completion call against a single model, with friendly error translation."""
    try:
        resp = _client().chat.completions.create(
            model=model, messages=messages, temperature=temperature,
            max_tokens=2500, timeout=90,
        )
    except AuthenticationError:
        raise LLMError("OpenRouter rejected the API key. Check OPENROUTER_API_KEY in .env.",
                       fatal=True)
    except APIConnectionError:
        raise LLMError("Could not reach OpenRouter. Check your internet connection.", fatal=True)
    except RateLimitError:
        raise LLMError(f"Rate limit hit on {model} (free models are throttled).")
    except APIStatusError as e:
        raise LLMError(f"OpenRouter returned HTTP {e.status_code} for {model}.")
    if not resp.choices or not resp.choices[0].message.content:
        raise LLMError(f"{model} returned an empty response.")
    return resp.choices[0].message.content


def _extract_json(raw: str) -> dict:
    """Pull a JSON object out of a reply, tolerating code fences / chatter."""
    text = re.sub(r"```(?:json)?", "", raw).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in reply")
    return json.loads(text[start:end + 1])


def call_llm_json(system: str, user: str, model_cls: type[BaseModel],
                  temperature: float = 0.4, attempts: int = 2) -> BaseModel:
    """Call the LLM and validate its JSON.

    For each model in MODELS: up to `attempts` tries (malformed JSON triggers a repair
    request). API errors or exhausted attempts fall through to the next model.
    """
    key = _cache_key(system, user, model_cls, temperature)
    if cache_enabled:
        hit = _cache_get(key)
        if hit is not None:
            try:
                result = model_cls.model_validate(hit)
                cache_stats["hits"] += 1
                return result
            except ValidationError:
                pass  # stale/corrupt entry -> fall through to a real call
        cache_stats["misses"] += 1

    last_err = "unknown error"
    for model in MODELS:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        for _ in range(attempts):
            try:
                raw = _chat(messages, temperature, model)
            except LLMError as e:
                if e.fatal:
                    raise
                last_err = str(e)
                break  # this model is unavailable -> try the next one
            try:
                result = model_cls.model_validate(_extract_json(raw))
                if cache_enabled:
                    _cache_put(key, result.model_dump())
                return result
            except (ValueError, ValidationError) as e:  # JSONDecodeError is a ValueError
                last_err = f"{model} returned malformed JSON ({str(e)[:120]})"
                messages += [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": f"That reply was invalid ({str(e)[:300]}). "
                                                "Return ONLY corrected JSON matching the schema."},
                ]
    raise LLMError(f"All configured models failed. Last error: {last_err}. "
                   "Wait a moment and retry, or add more models to OPENROUTER_MODEL.")


def _norm(topic: str) -> str:
    return topic.strip().title() or "General"


def aggregate_topic_scores(results: list[dict]) -> dict[str, float]:
    """Average score per topic. results = [{'topic', 'score', ...}]."""
    buckets: dict[str, list[int]] = {}
    for r in results:
        buckets.setdefault(_norm(r["topic"]), []).append(r["score"])
    return {t: round(sum(s) / len(s), 1) for t, s in buckets.items()}


# --------------------------------------------------------------------------- #
# Graph 1: Resume + JD -> Questions
# --------------------------------------------------------------------------- #
class QuestionState(TypedDict, total=False):
    resume: str
    jd: str
    role: str
    num_questions: int
    profile: dict
    questions: list[dict]
    reviews: list[dict]
    weak_indices: list[int]
    revision_count: int


def extract_skills(state: QuestionState) -> dict:
    """Node 1: structured skills / gaps from resume + JD."""
    profile = call_llm_json(
        prompts.SKILL_SYSTEM,
        prompts.skill_user(state["resume"][:MAX_CHARS], state["jd"][:MAX_CHARS], state["role"]),
        SkillProfile, temperature=0.2)
    return {"profile": profile.model_dump(), "revision_count": 0, "weak_indices": []}


def generate_questions(state: QuestionState) -> dict:
    """Node 2: generate questions; on a revision pass, rewrite only the weak ones."""
    n = state["num_questions"]
    previous, feedback = None, None
    if state.get("weak_indices"):
        previous = state["questions"]
        feedback = [r for r in state["reviews"] if r["index"] in state["weak_indices"]]
    result = call_llm_json(
        prompts.QUESTION_SYSTEM,
        prompts.question_user(state["role"], n, state["profile"], state["resume"][:MAX_CHARS],
                              state["jd"][:MAX_CHARS], previous, feedback),
        QuestionSet, temperature=0.7)
    return {"questions": [q.model_dump() for q in result.questions[:n]]}


def review_questions(state: QuestionState) -> dict:
    """Node 3: score relevance of each question; flag weak ones for a rewrite."""
    qs = state["questions"]
    result = call_llm_json(
        prompts.REVIEW_SYSTEM,
        prompts.review_user(state["role"], state["jd"][:MAX_CHARS], state["profile"], qs),
        ReviewSet, temperature=0.1)
    reviews = [r.model_dump() for r in result.reviews if 0 <= r.index < len(qs)]
    weak = [r["index"] for r in reviews if r["score"] < RELEVANCE_THRESHOLD]
    return {"reviews": reviews, "weak_indices": weak,
            "revision_count": state.get("revision_count", 0) + 1}


def route_after_review(state: QuestionState) -> str:
    """Conditional edge: loop back to the generator once if weak questions remain."""
    if state.get("weak_indices") and state["revision_count"] <= MAX_REVISIONS:
        return "generate_questions"
    return END


def build_question_graph():
    g = StateGraph(QuestionState)
    g.add_node("extract_skills", extract_skills)
    g.add_node("generate_questions", generate_questions)
    g.add_node("review_questions", review_questions)
    g.add_edge(START, "extract_skills")
    g.add_edge("extract_skills", "generate_questions")
    g.add_edge("generate_questions", "review_questions")
    g.add_conditional_edges("review_questions", route_after_review,
                            {"generate_questions": "generate_questions", END: END})
    return g.compile()


# --------------------------------------------------------------------------- #
# Graph 2: Answer evaluator
# --------------------------------------------------------------------------- #
class EvalState(TypedDict, total=False):
    role: str
    question: str
    topic: str
    answer: str
    is_followup: bool        # True when evaluating a follow-up (prevents endless follow-ups)
    evaluation: dict
    followup: dict


def evaluate_answer(state: EvalState) -> dict:
    """Score one answer out of 10 with strengths, missed concepts and a model answer."""
    ev = call_llm_json(
        prompts.EVAL_SYSTEM,
        prompts.eval_user(state["role"], state["question"], state["topic"], state["answer"][:MAX_CHARS]),
        Evaluation, temperature=0.2)
    return {"evaluation": ev.model_dump()}


def generate_followup(state: EvalState) -> dict:
    """Adaptive node: probe the concepts a weak answer missed with one follow-up question."""
    fu = call_llm_json(
        prompts.FOLLOWUP_SYSTEM,
        prompts.followup_user(state["role"], state["question"], state["answer"][:MAX_CHARS],
                              state["evaluation"]["missed_concepts"]),
        FollowUp, temperature=0.6)
    return {"followup": {"question": fu.question, "topic": state["topic"]}}


def route_after_eval(state: EvalState) -> str:
    """Conditional edge: weak first-attempt answers get a follow-up question."""
    if not state.get("is_followup") and state["evaluation"]["score"] < FOLLOWUP_THRESHOLD:
        return "generate_followup"
    return END


def build_eval_graph():
    g = StateGraph(EvalState)
    g.add_node("evaluate_answer", evaluate_answer)
    g.add_node("generate_followup", generate_followup)
    g.add_edge(START, "evaluate_answer")
    g.add_conditional_edges("evaluate_answer", route_after_eval,
                            {"generate_followup": "generate_followup", END: END})
    g.add_edge("generate_followup", END)
    return g.compile()


# --------------------------------------------------------------------------- #
# Graph 3: Improvement plan
# --------------------------------------------------------------------------- #
class PlanState(TypedDict, total=False):
    role: str
    gaps: list[str]
    results: list[dict]      # [{'topic', 'score', 'missed': [...]}]
    topic_scores: dict
    plan: dict


def recommend(state: PlanState) -> dict:
    """Aggregate scores per topic, pick the 3 weakest, ask the LLM for resources + 7-day plan."""
    results = state["results"]
    topic_scores = aggregate_topic_scores(results)
    weakest = sorted(topic_scores.items(), key=lambda kv: kv[1])[:3]
    missed: dict[str, list[str]] = {}
    for r in results:
        missed.setdefault(_norm(r["topic"]), []).extend(r.get("missed", []))
    payload = [{"topic": t, "avg_score": s, "missed_concepts": missed.get(t, [])[:6]}
               for t, s in weakest]
    plan = call_llm_json(prompts.PLAN_SYSTEM,
                         prompts.plan_user(state["role"], payload, state.get("gaps", [])),
                         ImprovementPlan, temperature=0.5)
    plan.weak_topics = plan.weak_topics[:3]
    plan.study_plan.sort(key=lambda d: d.day)
    return {"topic_scores": topic_scores, "plan": plan.model_dump()}


def build_plan_graph():
    g = StateGraph(PlanState)
    g.add_node("recommend", recommend)
    g.add_edge(START, "recommend")
    g.add_edge("recommend", END)
    return g.compile()


# --------------------------------------------------------------------------- #
# Graph 4: Resume booster
# --------------------------------------------------------------------------- #
class ResumeState(TypedDict, total=False):
    resume: str
    jd: str
    role: str
    gaps: list[str]
    tips: dict


def _squash(text: str) -> str:
    """Normalise for fuzzy containment checks: drop bullets, collapse whitespace, lowercase."""
    return re.sub(r"\s+", " ", re.sub(r"^[\-\u2022*\s]+", "", text)).strip().lower()


def improve_resume(state: ResumeState) -> dict:
    """Rewrite weak resume bullets for the JD, keeping only rewrites whose 'original' really
    appears in the resume (guards against the model inventing bullets)."""
    tips = call_llm_json(
        prompts.RESUME_SYSTEM,
        prompts.resume_user(state["role"], state["resume"][:MAX_CHARS], state["jd"][:MAX_CHARS],
                            state.get("gaps", [])),
        ResumeTips, temperature=0.4)
    haystack = _squash(state["resume"])
    tips.rewrites = [r for r in tips.rewrites if _squash(r.original)[:40] in haystack][:5]
    return {"tips": tips.model_dump()}


def build_resume_graph():
    g = StateGraph(ResumeState)
    g.add_node("improve_resume", improve_resume)
    g.add_edge(START, "improve_resume")
    g.add_edge("improve_resume", END)
    return g.compile()


# --------------------------------------------------------------------------- #
# Diagram helper
# --------------------------------------------------------------------------- #
def graph_to_dot(compiled) -> str:
    """Convert a compiled LangGraph into a Graphviz DOT string (conditional edges dashed)."""
    g = compiled.get_graph()
    lines = [
        "digraph G {", "rankdir=TB;", 'bgcolor="transparent";',
        'node [shape=box, style="rounded,filled", fillcolor="#eef2ff", color="#6366f1", '
        'fontcolor="#1e293b", fontname="Helvetica", fontsize=12];',
        'edge [color="#64748b", fontname="Helvetica"];',
    ]
    for node_id in g.nodes:
        if node_id in ("__start__", "__end__"):
            label = "START" if node_id == "__start__" else "END"
            lines.append(f'"{node_id}" [label="{label}", shape=circle, '
                         'fillcolor="#e2e8f0", color="#64748b"];')
        else:
            lines.append(f'"{node_id}" [label="{node_id}"];')
    for e in g.edges:
        style = ' [style=dashed, color="#6366f1"]' if e.conditional else ""
        lines.append(f'"{e.source}" -> "{e.target}"{style};')
    lines.append("}")
    return "\n".join(lines)
