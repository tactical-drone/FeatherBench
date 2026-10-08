<!-- Part of nn-playground. AGPL-3.0; for other licensing see COMMERCIAL.md. -->

# The FeatherBench prompt (for bridgebench.ai/test-prompts)

**Prompt name:** Recursive Super Intelligence: Featherweight Championship of the world! Nerds vs AI, place your bets!

**Linked repository:** https://github.com/tactical-drone/FeatherBench

**Prompt content** (paste only what is between the lines; current standings are always in
`featherbench/leaderboard.json`). For a self-contained version that needs no repo, see
`docs/featherbench-oneshot-prompt.md`.

---

Welcome to the RSI (Recursive Super Intelligence) Featherweight Championship: you, an AI,
will design a smarter AI. You are going to try to top the FeatherBench Intelligence Index,
an open benchmark where the rule is simple: the fewest parameters that solve every pattern wins.
It tests everything: planning, code, tools, evolutionary search, compute and honest
statistics. A winning setup is a reusable discovery (the same job with less memory), your
result is recorded per AI model, and humans will try to beat it by hand. Nerds vs AI.

Repository: https://github.com/tactical-drone/FeatherBench

The challenge: 12 two-dimensional pattern families (rings, wedges, an egg crate, a
checkerboard, hex tiles, parity lines, Voronoi cells, a 3-arm spiral, a tiny cluster
hiding in blobs, the Mandelbrot set, yin-yang, a smiley), 3 seeds each, a fixed budget of
3000 training steps. A pattern is solved when the mean accuracy on 2000 fresh points is at
least 90%. The score is params_max, the parameter count of your net on its biggest
dataset. The top spot goes to whoever solves all 12 with the fewest params. Current
standings are in featherbench/leaderboard.json.

Time-box, so attempts are comparable across models: stop after 45 minutes of your own work
or 40 full benchmark runs, whichever comes first, and report your best setup even if it does
not solve all 12.

Work like this:

1. Clone the repo, create a Python 3.10+ virtual environment, run
   `pip install -r requirements.txt`, then read README.md (the FeatherBench section),
   featherbench/README.md (the rules) and AGENTS.md (the console, every lever, budgets,
   error codes). Everything is driven through `python -m rsi <command>`, which prints one
   JSON document per command.
2. Start from the best accepted setup: `python -m rsi featherbench start` writes the current
   leader's setup to my_setup.json and lists its weakest patterns. Score it with
   `python -m rsi featherbench --config my_setup.json --workers 4` and note which patterns
   fail and by how much. Diagnose before you change anything.
3. Improve one change at a time and judge every change over all 3 seeds, never one lucky
   run. Use `python -m rsi describe` to see what can be changed, `python -m rsi run
   --datasets "..." --seeds 0-2` for quick probes on the failing patterns, and
   `python -m rsi evolve --objective bench:featherbench-general-v1 --background` to
   search. Think about why a pattern fails (frequency, symmetry, a tiny minority class)
   before throwing parameters at it.
4. Once a setup solves all 12, shrink it: lower width, classes or layer sizes step by
   step while every pattern stays solved. Fewer parameters is the whole point.
5. Save the best setup as a settings file, record who made the attempt and what it took
   (your model name, the tool you ran in, total tokens and cost in USD if you can see
   them, otherwise leave them out; never guess):
   `python -m rsi featherbench stamp featherbench/submissions/<my GitHub username>.json
   --ai-model "<your model>" --harness "<your tool>" --tokens N --cost-usd X --human-assist none`
   Then check it exactly as CI will:
   `python -m rsi featherbench verify featherbench/submissions/<my GitHub username>.json
   --github <my GitHub username> --workers 4`
6. Show your work: save the decision map of your final setup for each of the 12 patterns
   (`python -m rsi run --config <your file> --datasets "<pattern>" --png <pattern>.png`) and
   tile them into one image, featherbench-<model>.png.
7. End with this scorecard, filled in exactly (write "unknown" for anything you cannot see):

       FEATHERBENCH SCORECARD
       model: <your model name and version>
       harness: <the tool you ran in>
       params_max: <n>    feather_score: <n>    solved: <n>/12    mean_acc: <x.x %>
       tokens: <n | unknown>    cost_usd: <x | unknown>    wall_minutes: <n>    benchmark_runs: <n>
       key insight: <one sentence: what made the difference>

   Then list the per-dataset accuracies, what you changed from the default and what you
   tried that did not work.
8. Do not open a pull request yourself. If the result beats the current leaderboard
   (featherbench/leaderboard.json), show me the submission file and the exact git and gh
   commands for the pull request, and I'll decide whether to submit it under my name.

Rules that matter: a submission is a JSON settings file only, never code; it may only use
parts that already exist in the repo; at most 20,000 params on any dataset. Budget your
compute: one full benchmark run takes about a minute on 4 CPU cores, and results are
cached, so repeating a run is free.

---
