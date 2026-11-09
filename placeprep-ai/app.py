"""PlacePrep AI - Streamlit front-end.

Run with:  streamlit run app.py
Flow: Input -> Practice -> Results (sidebar stepper).
"""
import datetime
import html
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import plotly.graph_objects as go
import streamlit as st
from pypdf import PdfReader

import graph as G
import prompts
from graph import (FOLLOWUP_THRESHOLD, MAX_REVISIONS, MODELS, LLMError, aggregate_topic_scores,
                   build_eval_graph, build_plan_graph, build_question_graph,
                   build_resume_graph, graph_to_dot)

st.set_page_config(page_title="PlacePrep AI", page_icon="🎯", layout="wide")

STEPS = ["1 · Input", "2 · Practice", "3 · Results"]
ROLES = ["Backend Developer", "AI/ML Engineer", "Full-Stack Developer",
         "Data Analyst", "DevOps Engineer", "Software Engineer (SDE)"]
MAX_WORKERS = 3  # parallel evaluation threads (kept low for free-tier rate limits)


@st.cache_resource
def get_graphs():
    """Compile the three LangGraph pipelines once per server process."""
    return {"Question pipeline": build_question_graph(),
            "Answer evaluator": build_eval_graph(),
            "Improvement planner": build_plan_graph(),
            "Resume booster": build_resume_graph()}


GRAPHS = get_graphs()

# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
DEFAULTS = {"step": STEPS[0], "resume_text": "", "jd_text": "", "role": ROLES[0],
            "n_questions": 5, "qs_result": None, "evals": {}, "followup_scores": {},
            "plan_result": None, "resume_tips": None}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

# Apply a pending navigation request BEFORE the sidebar radio is created.
if "_goto" in st.session_state:
    st.session_state["step"] = st.session_state.pop("_goto")

# Streamlit drops widget-bound keys when the widget is not rendered on a page;
# re-assigning them keeps user input alive while moving between steps.
for _k in list(st.session_state.keys()):
    if _k in ("resume_text", "jd_text", "role", "n_questions") or _k.startswith(("ans_", "fu_")):
        st.session_state[_k] = st.session_state[_k]


def goto(step: str):
    """Request navigation (applied on the next rerun)."""
    st.session_state["_goto"] = step


def load_demo():
    """Button callback: fill inputs with demo data."""
    st.session_state.update(resume_text=prompts.DEMO_RESUME, jd_text=prompts.DEMO_JD,
                            role=prompts.DEMO_ROLE)


def on_pdf_upload():
    """File-uploader callback: extract PDF text into the resume box."""
    f = st.session_state.get("resume_pdf")
    if f is None:
        return
    try:
        text = "\n".join((p.extract_text() or "") for p in PdfReader(f).pages).strip()
    except Exception as e:
        st.session_state["_pdf_msg"] = ("error", f"Could not read that PDF: {e}")
        return
    if not text:
        st.session_state["_pdf_msg"] = ("error", "No text found (scanned PDF?). Please paste the text instead.")
        return
    st.session_state.resume_text = text
    st.session_state["_pdf_msg"] = ("ok", f"Loaded {len(text):,} characters from {f.name}.")


# --------------------------------------------------------------------------- #
# UI helpers
# --------------------------------------------------------------------------- #
def score_color(score: float) -> str:
    """Green >= 7, amber 4-6.9, red < 4."""
    return "#16a34a" if score >= 7 else "#d97706" if score >= 4 else "#dc2626"


def score_card(title: str, subtitle: str, score: float) -> str:
    c = score_color(score)
    return (f"<div style='border-left:6px solid {c};background:{c}1a;padding:12px 16px;"
            f"border-radius:8px;margin-bottom:10px'>"
            f"<div style='font-size:2rem;font-weight:700;color:{c}'>{score:g}"
            f"<span style='font-size:1rem'>/10</span></div>"
            f"<div style='font-weight:600'>{html.escape(title)}</div>"
            f"<div style='opacity:.7;font-size:.85rem'>{html.escape(subtitle)}</div></div>")


def short(text: str, n: int = 90) -> str:
    return text if len(text) <= n else text[:n].rstrip() + "…"


def radar_chart(topic_scores: dict) -> go.Figure:
    topics, vals = list(topic_scores), list(topic_scores.values())
    fig = go.Figure(go.Scatterpolar(r=vals + vals[:1], theta=topics + topics[:1],
                                    fill="toself", line_color="#6366f1",
                                    fillcolor="rgba(99,102,241,0.3)"))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 10], tickvals=[2, 4, 6, 8, 10])),
                      showlegend=False, margin=dict(l=40, r=40, t=30, b=30), height=420)
    return fig


def show_error(e: Exception):
    st.error(f"⚠️ {e}")


def evaluate_parallel(jobs: dict, label: str):
    """Run the evaluator graph for many answers concurrently.

    jobs: {key: graph_input}. Returns ({key: graph_output}, first_error_or_None);
    successful results are kept even if some calls fail.
    """
    results, error = {}, None
    bar = st.progress(0.0, text=label)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(GRAPHS["Answer evaluator"].invoke, inp): key
                   for key, inp in jobs.items()}
        for n, fut in enumerate(as_completed(futures), 1):
            try:
                results[futures[fut]] = fut.result()
            except Exception as e:  # LLMError or unexpected; keep the first one
                error = error or e
            bar.progress(n / len(jobs), text=f"{label} {n}/{len(jobs)}")
    bar.empty()
    return results, error


def build_report(role, overall, topic_scores, questions, evals, followup_scores, plan,
                 resume_tips=None) -> str:
    """Assemble a Markdown report of the whole session."""
    out = [f"# PlacePrep AI Report: {role}", f"_Generated {datetime.date.today().isoformat()}_", "",
           f"**Overall score:** {overall}/10", "", "## Topic scores", "",
           "| Topic | Avg score |", "|---|---|"]
    out += [f"| {t} | {v}/10 |" for t, v in sorted(topic_scores.items(), key=lambda kv: kv[1])]
    out += ["", "## Question feedback"]
    for i, e in sorted(evals.items()):
        q = questions[i]
        out += ["", f"### Q{i + 1}. {q['question']}", f"*{q['topic']} · {q['difficulty']}* · **Score: {e['score']}/10**",
                "", "**Your answer**", "", e["answer"], "", "**Strengths**"]
        out += [f"- {x}" for x in e["strengths"]] or ["- –"]
        out += ["", "**Missed concepts**"] + ([f"- {x}" for x in e["missed_concepts"]] or ["- –"])
        out += ["", "**Model answer**", "", e["model_answer"]]
        if e.get("followup"):
            out += ["", f"**Follow-up:** {e['followup']['question']}"]
            fs = followup_scores.get(i)
            if fs:
                out += [f"- Your answer: {fs['answer']}", f"- Score: {fs['score']}/10"]
    if plan:
        out += ["", "## Top weak topics"]
        for wt in plan["weak_topics"]:
            out += ["", f"### {wt['topic']}", wt["why"], ""] + [f"- {r}" for r in wt["resources"]]
        out += ["", "## 7-day study plan"]
        for d in plan["study_plan"]:
            out += ["", f"**Day {d['day']}: {d['focus']}**"] + [f"- {t}" for t in d["tasks"]]
    if resume_tips:
        out += ["", "## Resume suggestions"]
        if resume_tips["summary"]:
            out += ["", f"**Suggested summary:** {resume_tips['summary']}"]
        if resume_tips["missing_keywords"]:
            out += ["", "**Keywords to add (only where true):** " + ", ".join(resume_tips["missing_keywords"])]
        for r in resume_tips["rewrites"]:
            out += ["", f"- **Before:** {r['original']}", f"  - **After:** {r['improved']}"]
    return "\n".join(out) + "\n"


def describe_step(node: str, update: dict, state: dict, secs: float) -> str:
    """Human-friendly progress line for a finished node of the question pipeline."""
    t = f" _({secs:.0f}s)_"
    if node == "extract_skills":
        n = len(update["profile"]["skills"])
        return f"✅ **Skills extracted** from resume & JD ({n} skills found){t}"
    if node == "generate_questions":
        revised = state.get("revision_count", 0) > 0
        return f"✅ **Questions {'revised' if revised else 'drafted'}** ({len(update['questions'])}){t}"
    weak = update["weak_indices"]
    if not weak:
        return f"✅ **Reviewer approved** all questions{t}"
    if update["revision_count"] <= MAX_REVISIONS:
        return f"🔁 **Reviewer flagged {len(weak)} weak question(s)**, sending back to the generator{t}"
    return f"✅ **Review finished**: {len(weak)} question(s) still borderline, keeping best effort{t}"


def run_question_pipeline(inputs: dict) -> dict:
    """Stream the question graph so the user sees each node finish, and return the final state."""
    state = dict(inputs)
    with st.status("Building your questions…", expanded=True) as status:
        try:
            last = time.perf_counter()
            for chunk in GRAPHS["Question pipeline"].stream(inputs, stream_mode="updates"):
                for node, update in chunk.items():
                    state.update(update)
                    now = time.perf_counter()
                    st.write(describe_step(node, update, state, now - last))
                    last = now
        except Exception:
            status.update(label="Something went wrong", state="error")
            raise
        status.update(label="Questions ready", state="complete", expanded=False)
    return state


# --------------------------------------------------------------------------- #
# Sidebar stepper
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("🎯 PlacePrep AI")
    st.radio("Steps", STEPS, key="step")
    st.divider()
    has_key = bool(os.getenv("OPENROUTER_API_KEY", "").strip())
    st.caption("Models: " + " → ".join(f"`{m}`" for m in MODELS))
    st.caption("API key: ✅ loaded" if has_key else "API key: ❌ missing (see .env.example)")
    G.cache_enabled = st.toggle("Response cache", value=G.cache_enabled,
                                help="Identical prompts return instantly instead of calling the API.")
    st.caption(f"Cache: {G.cache_stats['hits']} hits · {G.cache_stats['misses']} misses")
    if st.button("Reset session"):
        st.session_state.clear()
        st.rerun()

step = st.session_state["step"]
qs_result = st.session_state.qs_result


# --------------------------------------------------------------------------- #
# Step 1: Input
# --------------------------------------------------------------------------- #
def page_input():
    st.title("🎯 PlacePrep AI")
    st.markdown("**Your AI placement coach:** paste a resume + job description, practise "
                "tailored questions, get scored feedback, follow-up drills and a 7-day plan.")

    with st.expander("🧠 How it works: LangGraph pipelines", expanded=True):
        cols = st.columns([4, 3, 2, 2])
        for col, (name, g) in zip(cols, GRAPHS.items()):
            with col:
                st.markdown(f"**{name}**")
                try:
                    st.graphviz_chart(graph_to_dot(g))  # rendered in-browser, no CDN needed
                except Exception as e:  # diagram is cosmetic, never block the app
                    st.caption(f"Diagram unavailable: {e}")
        st.caption(f"Dashed arrows = conditional edges. The reviewer sends weak questions back to the "
                   f"generator (max 1 loop); answers scoring below {FOLLOWUP_THRESHOLD} get a follow-up question.")

    st.button("✨ Load Demo Data", on_click=load_demo)
    st.file_uploader("📄 Or upload your resume as PDF (optional)", type="pdf",
                     key="resume_pdf", on_change=on_pdf_upload)
    kind, msg = st.session_state.get("_pdf_msg", (None, None))
    if kind == "error":
        st.error(msg)
    elif kind == "ok":
        st.success(msg)

    left, right = st.columns(2)
    left.text_area("Resume", key="resume_text", height=280, placeholder="Paste your resume text…")
    right.text_area("Job description", key="jd_text", height=280, placeholder="Paste the JD…")
    c1, c2, _ = st.columns([2, 2, 3])
    c1.selectbox("Target role", ROLES, key="role")
    c2.slider("Number of questions", 3, 8, key="n_questions")

    if st.button("🚀 Generate Questions", type="primary"):
        if not st.session_state.resume_text.strip() or not st.session_state.jd_text.strip():
            st.warning("Please provide both a resume and a job description (or load the demo data).")
            return
        try:
            out = run_question_pipeline({
                "resume": st.session_state.resume_text, "jd": st.session_state.jd_text,
                "role": st.session_state.role, "num_questions": st.session_state.n_questions})
        except LLMError as e:
            show_error(e)
            return
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            return
        # Fresh question set -> clear previous answers / results.
        for k in [k for k in st.session_state if k.startswith(("ans_", "fu_"))]:
            del st.session_state[k]
        st.session_state.update(qs_result=out, evals={}, followup_scores={}, plan_result=None,
                                resume_tips=None)
        goto(STEPS[1])
        st.rerun()


# --------------------------------------------------------------------------- #
# Step 2: Practice
# --------------------------------------------------------------------------- #
def run_evaluation():
    """Evaluate answered questions in parallel (unchanged answers are not re-scored)."""
    questions = qs_result["questions"]
    answered = {i: st.session_state.get(f"ans_{i}", "").strip() for i in range(len(questions))}
    answered = {i: a for i, a in answered.items() if a}
    if not answered:
        st.warning("Answer at least one question first.")
        return
    # Keep evaluations whose answer is unchanged; re-score the rest.
    kept = {i: e for i, e in st.session_state.evals.items() if answered.get(i) == e["answer"]}
    jobs = {i: {"role": st.session_state.role, "question": questions[i]["question"],
                "topic": questions[i]["topic"], "answer": a}
            for i, a in answered.items() if i not in kept}
    new, err = evaluate_parallel(jobs, "Evaluating answers…") if jobs else ({}, None)
    evals = dict(kept)
    for i, out in new.items():
        evals[i] = {"answer": answered[i], **out["evaluation"], "followup": out.get("followup")}
    st.session_state.evals = evals
    # Follow-up scores only stay valid for evaluations that were not redone.
    st.session_state.followup_scores = {i: s for i, s in st.session_state.followup_scores.items()
                                        if i in kept}
    st.session_state.plan_result = None
    if err:
        show_error(err)
        st.caption("Successful evaluations were kept. Click again to retry the rest.")
        return
    goto(STEPS[2])
    st.rerun()


def run_resume_booster():
    try:
        with st.spinner("Rewriting weak bullets for this JD…"):
            out = GRAPHS["Resume booster"].invoke({
                "resume": st.session_state.resume_text, "jd": st.session_state.jd_text,
                "role": st.session_state.role, "gaps": qs_result["profile"].get("gaps", [])})
        st.session_state.resume_tips = out["tips"]
    except LLMError as e:
        show_error(e)


def show_resume_tips(tips):
    if not tips:
        return
    if tips["missing_keywords"]:
        st.markdown("**Keywords to add (only where true for you):** "
                    + " ".join(f"`{k}`" for k in tips["missing_keywords"]))
    if tips["summary"]:
        st.markdown("**Suggested summary line**")
        st.info(tips["summary"])
    if not tips["rewrites"]:
        st.caption("No bullet rewrites could be matched to your resume text. Try again.")
    for r in tips["rewrites"]:
        before, after = st.columns(2)
        before.markdown("**Before**")
        before.warning(r["original"])
        after.markdown("**After**")
        after.success(r["improved"])
        if r["why"]:
            st.caption(r["why"])
    st.caption("⚠️ Fill in placeholders like [X%] with your real numbers, and never claim anything untrue.")


def page_practice():
    st.header("✍️ Practice")
    if not qs_result:
        st.info("Generate questions in step 1 first.")
        return
    prof = qs_result["profile"]
    with st.expander("Detected skills & gaps"):
        st.markdown("**Skills:** " + (", ".join(prof["skills"]) or "–"))
        st.markdown("**Gaps vs JD:** " + (", ".join(prof["gaps"]) or "–"))
    relevance = {r["index"]: r for r in qs_result.get("reviews", [])}

    with st.expander("📝 Resume booster (optional)"):
        st.caption("Rewrites your weakest bullets to fit this job description, without inventing facts.")
        if st.button("Improve my resume bullets"):
            run_resume_booster()
        show_resume_tips(st.session_state.resume_tips)

    total = len(qs_result["questions"])
    done = sum(1 for i in range(total) if st.session_state.get(f"ans_{i}", "").strip())
    st.progress(done / total, text=f"{done} of {total} answered "
                                   "(press Ctrl+Enter or click outside a box to save an answer)")

    for i, q in enumerate(qs_result["questions"]):
        with st.container(border=True):
            tick = "✅ " if st.session_state.get(f"ans_{i}", "").strip() else ""
            st.markdown(f"**{tick}Q{i + 1}. {q['question']}**")
            rel = relevance.get(i)
            st.caption(f"{q['topic']} · {q['difficulty']}"
                       + (f" · reviewer relevance {rel['score']}/10" if rel else ""))
            st.text_area("Your answer", key=f"ans_{i}", height=120, label_visibility="collapsed",
                         placeholder="Type your answer…")
    if st.button("📊 Evaluate My Answers", type="primary"):
        run_evaluation()


# --------------------------------------------------------------------------- #
# Step 3: Results
# --------------------------------------------------------------------------- #
def run_followup_evaluation(pending: dict):
    """Score the follow-up answers (follow-ups never spawn further follow-ups)."""
    answered = {i: st.session_state.get(f"fu_{i}", "").strip() for i in pending}
    answered = {i: a for i, a in answered.items() if a}
    if not answered:
        st.warning("Answer at least one follow-up first.")
        return
    kept = {i: s for i, s in st.session_state.followup_scores.items()
            if answered.get(i) == s["answer"]}
    jobs = {i: {"role": st.session_state.role, "question": pending[i]["question"],
                "topic": pending[i]["topic"], "answer": a, "is_followup": True}
            for i, a in answered.items() if i not in kept}
    new, err = evaluate_parallel(jobs, "Evaluating follow-ups…") if jobs else ({}, None)
    scores = dict(kept)
    for i, out in new.items():
        scores[i] = {"answer": answered[i], **out["evaluation"]}
    st.session_state.followup_scores = scores
    st.session_state.plan_result = None  # scores changed -> plan is outdated
    if err:
        st.session_state["_flash"] = str(err)
    st.rerun()


def page_results():
    st.header("📊 Results")
    flash = st.session_state.pop("_flash", None)
    if flash:
        show_error(flash)
    evals = st.session_state.evals
    followup_scores = st.session_state.followup_scores
    if not qs_result or not evals:
        st.info("Answer and evaluate questions in step 2 first.")
        return
    questions = qs_result["questions"]

    # Every scored attempt (first answers + follow-ups) feeds the aggregation.
    rows = [{"topic": questions[i]["topic"], "score": e["score"], "missed": e["missed_concepts"]}
            for i, e in evals.items()]
    rows += [{"topic": questions[i]["topic"], "score": s["score"], "missed": s["missed_concepts"]}
             for i, s in followup_scores.items() if i in evals]
    topic_scores = aggregate_topic_scores(rows)
    overall = round(sum(r["score"] for r in rows) / len(rows), 1)

    m1, m2, m3 = st.columns(3)
    m1.metric("Overall score", f"{overall}/10")
    m2.metric("Strongest topic", max(topic_scores, key=topic_scores.get))
    m3.metric("Weakest topic", min(topic_scores, key=topic_scores.get))

    # Colour-coded score cards, three per row.
    items = sorted(evals.items())
    for start in range(0, len(items), 3):
        for col, (i, e) in zip(st.columns(3), items[start:start + 3]):
            col.markdown(score_card(f"Q{i + 1}: {questions[i]['topic']}",
                                    short(questions[i]["question"]), e["score"]),
                         unsafe_allow_html=True)

    # Adaptive follow-ups for weak answers.
    pending = {i: e["followup"] for i, e in items if e.get("followup")}
    if pending:
        st.subheader("🔁 Follow-up drills")
        st.caption(f"These answers scored below {FOLLOWUP_THRESHOLD}. Try a focused follow-up on what was missed.")
        for i, fu in pending.items():
            with st.container(border=True):
                st.markdown(f"**Follow-up to Q{i + 1}:** {fu['question']}")
                st.text_area("Your answer", key=f"fu_{i}", height=100, label_visibility="collapsed",
                             placeholder="Type your answer…")
                fs = followup_scores.get(i)
                if fs:
                    st.markdown(score_card("Follow-up score", short(fs["missed_concepts"][0], 80)
                                           if fs["missed_concepts"] else "No major gaps", fs["score"]),
                                unsafe_allow_html=True)
                    with st.expander("Model answer"):
                        st.info(fs["model_answer"])
        if st.button("Evaluate follow-ups"):
            run_followup_evaluation(pending)

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Skill radar")
        st.plotly_chart(radar_chart(topic_scores))
        if len(topic_scores) < 3:
            st.caption("Answer questions from 3+ topics for a fuller radar.")
    with right:
        st.subheader("Question feedback")
        for i, e in items:
            with st.expander(f"Q{i + 1} · {questions[i]['topic']} · {e['score']}/10"):
                st.markdown("**Your answer**")
                st.write(e["answer"])
                st.markdown("**✅ Strengths**\n" + "\n".join(f"- {s}" for s in e["strengths"] or ["–"]))
                st.markdown("**❌ Missed concepts**\n" + "\n".join(f"- {s}" for s in e["missed_concepts"] or ["–"]))
                st.markdown("**💡 Model answer**")
                st.info(e["model_answer"])

    st.divider()
    st.subheader("🗺️ Improvement plan")
    if st.button("Generate improvement plan", type="primary"):
        try:
            with st.spinner("Building your plan…"):
                out = GRAPHS["Improvement planner"].invoke({
                    "role": st.session_state.role, "results": rows,
                    "gaps": qs_result["profile"].get("gaps", [])})
            st.session_state.plan_result = out
        except LLMError as e:
            show_error(e)
    plan = (st.session_state.plan_result or {}).get("plan")
    if plan:
        st.markdown("#### Top weak topics")
        for col, wt in zip(st.columns(len(plan["weak_topics"])), plan["weak_topics"]):
            avg = topic_scores.get(wt["topic"].strip().title(), 0)
            with col:
                st.markdown(score_card(wt["topic"], wt["why"], avg), unsafe_allow_html=True)
                st.markdown("**Resources**\n" + "\n".join(f"- {r}" for r in wt["resources"]))
        st.markdown("#### 7-day study plan")
        for d in plan["study_plan"]:
            with st.expander(f"Day {d['day']} · {d['focus']}", expanded=d["day"] == 1):
                st.markdown("\n".join(f"- {t}" for t in d["tasks"]))

    st.divider()
    st.download_button("📥 Download report (Markdown)",
                       data=build_report(st.session_state.role, overall, topic_scores, questions,
                                         evals, followup_scores, plan,
                                         st.session_state.resume_tips),
                       file_name="placeprep_report.md", mime="text/markdown")
    st.caption("The report includes the improvement plan once you've generated it.")


{STEPS[0]: page_input, STEPS[1]: page_practice, STEPS[2]: page_results}[step]()
