<!-- Part of nn-playground. AGPL-3.0; for other licensing see COMMERCIAL.md. -->

# The FeatherBench prompt (for bridgebench.ai/test-prompts)

**Prompt name:** Top the FeatherBench Intelligence Index: the smallest neural net that solves every pattern

**Linked repository:** https://github.com/tactical-drone/FeatherBench

**Prompt content** (everything between the lines; this is the posted version, so its
numbers are from launch day; current standings are always in `featherbench/leaderboard.json`):

---

You are going to try to top the FeatherBench Intelligence Index, an open benchmark where the rule
is simple: the fewest parameters that solve every pattern wins.

Repository: https://github.com/tactical-drone/FeatherBench

The challenge: 12 two-dimensional pattern families (rings, wedges, an egg crate, a
checkerboard, hex tiles, parity lines, Voronoi cells, a 3-arm spiral, a tiny cluster
hiding in blobs, the Mandelbrot set, yin-yang, a smiley), 3 seeds each, a fixed budget of
3000 training steps. A pattern is solved when the mean accuracy on 2000 fresh points is at
least 90%. The score is params_max, the parameter count of your net on its biggest
dataset. The top spot goes to whoever solves all 12 with the fewest params. At the time of
writing nobody has solved all 12; the house entry solves 9 at 346 params.

Work like this:

1. Clone the repo, create a Python 3.10+ virtual environment, run
   `pip install -r requirements.txt`, then read README.md (the FeatherBench section),
   featherbench/README.md (the rules) and AGENTS.md (the console, every lever, budgets,
   error codes). Everything is driven through `python -m rsi <command>`, which prints one
   JSON document per command.
2. Get the baseline: `python -m rsi featherbench --workers 4` scores the default setup.
   Note which datasets fail and by how much.
3. Improve one change at a time and judge every change over all 3 seeds, never one lucky
   run. Use `python -m rsi describe` to see what can be changed, `python -m rsi run
   --datasets "..." --seeds 0-2` for quick probes on the failing patterns, and
   `python -m rsi evolve --objective bench:featherbench-general-v1 --background` to
   search. Think about why a pattern fails (frequency, symmetry, a tiny minority class)
   before throwing parameters at it.
4. Once a setup solves all 12, shrink it: lower width, classes or layer sizes step by
   step while every pattern stays solved. Fewer parameters is the whole point.
5. Save the best setup as a settings file and check it exactly as CI will:
   `python -m rsi featherbench verify featherbench/submissions/<my GitHub username>.json
   --github <my GitHub username> --workers 4`
6. Report back to me: the final params_max, how many patterns it solves, the per-dataset
   accuracies, what you changed from the default, and what you tried that did not work.
7. Do not open a pull request yourself. If the result beats the current leaderboard
   (featherbench/leaderboard.json), show me the submission file and the exact git and gh
   commands for the pull request, and I'll decide whether to submit it under my name.

Rules that matter: a submission is a JSON settings file only, never code; it may only use
parts that already exist in the repo; at most 20,000 params on any dataset. Budget your
compute: one full benchmark run takes about a minute on 4 CPU cores, and results are
cached, so repeating a run is free.

---
