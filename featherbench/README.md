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

Ties go to the earlier submission. The **world record** is rank 1 among the setups that
solve all 12.

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
