<!-- Part of nn-playground. AGPL-3.0; for other licensing see COMMERCIAL.md. -->

# FeatherBench for NerfBench (and any model benchmark)

**Recursive Super Intelligence: Featherweight Championship of the world! Nerds vs AI, place
your bets!**

NerfBench tracks each model's power over time against its launch. FeatherBench adds one
more axis to it: **how well a model improves intelligence**, the core skill of recursive
self-improvement (RSI). The task: design a tiny neural net that solves 12 pattern families
(rings, wedges, an egg crate, a checkerboard, hex tiles, parity lines, Voronoi cells, a
spiral, a tiny cluster in blobs, the Mandelbrot set, yin-yang, a smiley) with as few
parameters as possible. It's a dark art: blindly adding width or classes often makes a
pattern worse. That's what makes it a good test.

## Why it fits

| NerfBench needs | FeatherBench one-shot gives |
|---|---|
| A fixed test, same every run | One frozen prompt and a frozen benchmark (definition hash `6e9b8a2dc8a4`) |
| Automatic scoring | `feather_score` 0-1000, no human judging |
| Deterministic | Same answer, same score on the same machine and code |
| Cheap | One reply per model; about 1.5 min of CPU to score (5 cores) |
| Something to look at | `--answer-png`: the 12 decision maps the model's net learned |
| Hard to saturate | Solving all 12 is reachable, then it's a race for fewer params (log scale) |

## Drop-in recipe

1. Each run, send every model the prompt in
   [featherbench-oneshot-prompt.md](featherbench-oneshot-prompt.md) (the part between the
   lines) and save the raw reply as `<model>.txt`.
2. Score it:

       git clone https://github.com/tactical-drone/FeatherBench && cd FeatherBench
       pip install -r requirements.txt
       python -m rsi featherbench score-answer <model>.txt --workers 5 --answer-png <model>.png

3. Plot `feather_score` per model per day, next to the other NerfBench tests. The output is
   one JSON document (`featherbench/oneshot@1`): `valid`, `feather_score`, `n_solved`,
   `params_max`, `mean_acc`, `min_acc`, `per_dataset`, and a `reason` when the answer is
   invalid (no JSON, an unknown part, more than 20,000 params), which scores 0.

Pin the repo to a release tag (e.g. `v0.3.0`) so the scorer never changes under you.

## The agentic variant

For models with tools: [featherbench-prompt.md](featherbench-prompt.md) lets an agent work
in the repo with the `rsi` console, including an evolutionary search, and submit a pull
request. It tests everything (planning, code, tools, compute, honest statistics), and each
leaderboard entry records the AI model, tokens and cost (self-reported), with a per-model
table. Slower and costlier than one-shot, but closer to how agents really work.

## What the compute buys

Every winning setup is a reusable discovery: a smaller, more general network that does the
same job with less memory. Anyone can load it in the FeatherBench app, check with their own
eyes that the decision maps match the numbers, and try to beat it by hand. Nerds vs AI.
