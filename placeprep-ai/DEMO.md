# PlacePrep AI: 3-minute demo script

## Before you go on stage (2 minutes)
- [ ] `.env` has your OpenRouter key and 2-3 `:free` models in `OPENROUTER_MODEL`.
- [ ] Do ONE full dry run with the demo data. This fills the response cache, so the live run is near-instant.
- [ ] Keep the app open on the **Input** page. Have a second browser tab with a saved copy of the Results page screenshot as a backup.
- [ ] Sidebar: **Response cache = ON** for the demo (turn it OFF only if you want to show fresh questions).

## The walkthrough

| Time | Click | Say |
|---|---|---|
| 0:00 | Home page, point at the 4 diagrams | "Four LangGraph pipelines. Dashed arrows are conditional edges: the reviewer can send weak questions back, and weak answers branch to a follow-up." |
| 0:25 | **Load Demo Data** → **Generate Questions** | "A resume and a JD go in. Watch the live steps: skills extracted, questions drafted, reviewed, and weak ones sent back once." |
| 0:55 | Practice page → open **Resume booster** → **Improve my resume bullets** | "It rewrites weak bullets for this JD but never invents facts. Unknown numbers become placeholders, and any rewrite that doesn't match the real resume is dropped." |
| 1:15 | Type short answers (make ONE deliberately weak) → **Evaluate My Answers** | "Answers are scored in parallel as structured JSON, validated with pydantic." |
| 1:45 | Results: score cards, radar chart | "Green, amber, red. The radar shows strength per topic." |
| 2:00 | **Follow-up drills** → answer the follow-up → **Evaluate follow-ups** | "The weak answer triggered an adaptive follow-up on exactly the concept it missed." |
| 2:25 | **Generate improvement plan** | "Top 3 weak topics, resources, and a 7-day plan." |
| 2:45 | **Download report** | "Everything exports as one Markdown report." |

**Tip:** answer 2 questions well and 1 badly, so you get a green card, a red card and a follow-up in one pass.

## If something goes wrong
- **Rate limit / JSON error:** the app already falls back to the next model. If it still fails, click the button again (cached steps are instant).
- **Total outage:** show the backup screenshot, then walk the diagrams and `graph.py` (the reviewer loop and the follow-up branch are the interesting code).
- **Diagrams blank:** they render offline, so just refresh the page.

## Why LangGraph? (30-second answer)
Questions need a quality loop: generate, review, regenerate only if weak. Answers need branching: a low score triggers a follow-up. Both are state machines with conditional edges, which is exactly what LangGraph models, and the graph can be drawn as the diagram on the home page.

## Likely judge questions
- **"How do you stop the model inventing things?"** Pydantic validation with one repair retry; resume rewrites must match the real resume text; placeholders for unknown numbers.
- **"What about prompt injection?"** Resume, JD and answers are wrapped in delimiter tags, and every prompt tells the model to treat them as data only.
- **"What if the free model is down?"** Comma-separated fallback models plus a response cache.
- **"What's next?"** Rubric scoring (correctness, depth, clarity), progress history, company-specific mode, live deployment.
