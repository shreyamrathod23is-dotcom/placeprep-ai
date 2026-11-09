# 🎯 PlacePrep AI

AI placement-preparation assistant built with **LangGraph + Streamlit + OpenRouter (free models) + Plotly**.

## Features
1. **Resume + JD → Questions**: `extract_skills → generate_questions → review_questions`. The reviewer scores relevance and sends weak questions back to the generator (conditional edge, max 1 loop). Resume can be pasted or uploaded as a **PDF**.
2. **Answer Evaluator**: score /10, strengths, missed concepts and a model answer as validated JSON. Answers are scored **in parallel**. Answers scoring below 5 trigger an **adaptive follow-up question** (conditional edge) that you answer on the Results page.
3. **Resume booster**: rewrites weak resume bullets for the JD (never inventing facts) and lists missing keywords.
4. **Improvement Plan**: aggregates scores per topic, returns the top 3 weak topics, resources and a 7-day plan, plus a Plotly radar chart. **Download everything as a Markdown report.**

## Reliability & safety
- Every LLM reply is validated with pydantic (one automatic repair retry); API errors show friendly messages.
- `OPENROUTER_MODEL` accepts a **comma-separated list of fallback models**.
- **Response cache** (`.cache/llm_cache.json`): identical prompts return instantly, so repeating a demo doesn't burn rate-limited calls. Toggle it in the sidebar or set `LLM_CACHE=0`. Delete `.cache/` to clear it.
- **Prompt-injection guard**: resume, JD and answers are wrapped in delimiter tags and the model is told to treat them as data only.
- Diagrams render with Streamlit's built-in Graphviz, so they work **offline**.

## Run

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # Windows: copy .env.example .env  -> then add your OpenRouter key
streamlit run app.py
```

Open http://localhost:8501, click **Load Demo Data**, then **Generate Questions**.

## Notes
- Free models are rate-limited; list a few in `OPENROUTER_MODEL` so the app can fall back automatically. Free model names change, so pick current `:free` ones at https://openrouter.ai/models.
- Scanned (image-only) PDFs have no extractable text; paste the text instead.
- Because of the cache, re-generating with the same inputs returns the same questions. Turn the cache off in the sidebar for fresh ones.
- See `DEMO.md` for a 3-minute demo script and likely judge questions.
- Files: `app.py` (UI), `graph.py` (LangGraph, LLM client, cache, validation), `prompts.py` (prompts + demo data).
