<!-- Part of nn-playground. AGPL-3.0; for other licensing see COMMERCIAL.md. -->

# FeatherBench rules

**Fewest params that solves every pattern wins.**

## The benchmark

`featherbench-general-v1` (alias `general-v1`, definition hash `6e9b8a2dc8a4`). Twelve
datasets, seeds 0-2, 600 points, noise 0.05, a random 80/20 split, 3000 training steps,
scored on 2000 fresh points per cell:

Bullseye (5 rings), Sectors (8 wedges), Egg crate (3), Checkerboard, Hex tiling (3, coarse),
Random lines (parity), Voronoi (6), Spiral (3 arms), Blobs + tiny cluster, Mandelbrot,
Yin-yang, Smiley.

A dataset is **solved** when the mean fresh accuracy over its 3 seeds is at least 90 %.
`params_max` is your net's parameter count on its biggest dataset (class counts differ).
The benchmark overrides the data fields of your config (points, noise, split, dataset,
seed); everything else in the config is your setup.

## Ranking

1. Solves all 12, fewest `params_max` first, then higher mean accuracy.
2. Then everyone else: most datasets solved, then mean accuracy, then fewer params.

Ties go to the earlier submission. The **top of the FeatherBench Intelligence Index** is rank 1 among
the setups that solve all 12.

## Submitting

1. Save your setup: File > Save settings in the app, or `python -m rsi export RUN --rank 1 -o file.json`.
2. Check it locally (optional, about a minute on 4 cores):
   `python -m rsi featherbench verify featherbench/submissions/YOU.json --github YOU --workers 4`
3. Open a pull request that adds or changes exactly one file:
   `featherbench/submissions/<your GitHub username>.json`. One entry per person; a new
   attempt replaces your old one.

A submission is a settings file (`nn-playground/settings`) or a bare config object, at
most 64 KB, using only fields and parts that exist in this repo, with at most 20,000
params on any dataset. It never contains or names code. Want a new activation, layer or
model? Contribute it as a normal pull request first; once merged, everyone can use it.

## Attempt details (model, tokens, cost)

Say who found the setup and what it took, in the settings file's `meta.attempt`:

    python -m rsi featherbench stamp featherbench/submissions/YOU.json       --ai-model "Claude Opus 5.5" --harness "Claude Code" --tokens 1200000 --cost-usd 4.20 --human-assist none

`model` is the AI that drove the attempt (or `human`), `harness` the tool it ran in,
`tokens` the total for the attempt, `cost_usd` what it cost, `human_assist` none / some /
lots. All optional, all **self-reported**: CI can't check them, and the leaderboard says
so. The leaderboard also has a `models` table: each AI model's best entry, how many
attempts, and their median tokens and cost. That's the other half of FeatherBench: it
ranks the nets and, through the attempts, the AIs that design them.

Every entry also gets a `feather_score` from 0 to 1000 that orders exactly like the
ranking (below 923: not every pattern solved; 923-1000: all 12 solved, fewer params higher).

## One-shot mode (no tools, no repo)

`docs/featherbench-oneshot-prompt.md` is a self-contained prompt any model can answer in
one reply with one JSON setup; `python -m rsi featherbench score-answer reply.txt`
scores the reply deterministically (0 with a reason if it's invalid). It's built for model
benchmarks that re-run fixed tests over time.

## How scoring stays honest

- The pull request check runs on `pull_request` with a read-only token and no secrets. It
  verifies the file and re-runs the benchmark so you can see your score, but it can't
  write anything.
- After a maintainer merges, `featherbench-leaderboard.yml` scores the file again on main,
  on GitHub's Linux runners with CPU torch 2.14, and only that run's numbers go into
  `results/` and `leaderboard.json`. Your local numbers may differ slightly (other
  platforms round differently); CI's numbers are the official ones.
- Every result records the benchmark hash, the full config, the platform and the code
  fingerprint, so anyone can reproduce it with `python -m rsi featherbench verify`.

The leaderboard's top 10 can be loaded straight into the app: FeatherBench > Top 10, pick
a racer, and try to beat their setup.
