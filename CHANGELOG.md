# Changelog

<!-- Part of nn-playground. AGPL-3.0; for other licensing see COMMERCIAL.md. -->

Versions follow `nncore.__version__` and `rsi.__version__`. Results JSON records both,
plus a code fingerprint, so you can tell which version produced a number.

## Unreleased

## 0.4.1 (2026-10-08)

- The RSI-Index baseline is frozen per benchmark version (`featherbench/baselines/`, CI's
  Linux numbers for the default setup), so the index stays comparable when the default
  changes and `verify` no longer re-scores the default on every check. The compute ledger is
  labelled self-reported: it travels in the editable submission file.

## 0.4.0 (2026-10-08)

- Held-out seeds: CI's leaderboard job also scores every merged submission on secret seeds
  (the `FEATHERBENCH_HOLDOUT_SEEDS` Actions secret, readable only by that job). Results and
  the leaderboard carry `holdout` (patterns solved, mean and worst accuracy; never the seeds)
  and an `overfit` flag; the held-out run logs nothing. The app's Top 10 shows a held-out column.
- An RSI loop after the RSIGym paper (docs/rsigym.md). `rsi featherbench start` inherits the
  current leader's setup (or `--inherit default` / a racer) and lists its weakest patterns;
  `stamp` records the parent and a compute ledger (trials, cells, CPU seconds since `start`)
  measured by the console itself. Results gain an `rsi_index` (mean share of the remaining
  accuracy gap closed over the default setup, per pattern) and the leaderboard a
  `gain_over_parent` per entry (recursive progress) and `best_rsi_index` per model. The
  agent loop in AGENTS.md now follows the paper's lessons: diagnose first, light changes,
  inherit what works.
## 0.3.0 (2026-10-07)

- The RSI release. RSI = Recursive Super Intelligence: intelligence that improves the next
  intelligence. FeatherBench is now billed as **Recursive Super Intelligence: Featherweight Championship of the world! Nerds vs AI, place your bets!**
  The README, both BridgeBench prompts, AGENTS.md, CLAUDE.md, the rsi console and the app
  (window title "FeatherBench // RSI playground", About box) use the name. No scoring changes.

## 0.2.0 (2026-10-07)

- FeatherBench ranks the AIs too. Submissions can carry a self-reported `meta.attempt`
  (AI model, harness, tokens, cost in USD, human assist), written with
  `rsi featherbench stamp`; the leaderboard shows it per entry, adds a `models` table
  (best entry, attempts, median tokens and cost per model) and a `feather_score` (0-1000,
  ordered like the ranking). The app's Top 10 shows the AI model, tokens and cost.
- One-shot mode: `docs/featherbench-oneshot-prompt.md`, a self-contained prompt answered
  with one JSON setup, and `rsi featherbench score-answer` to score a reply
  deterministically (0 with a reason when invalid; `--answer-png` draws the 12 maps).

## 0.1.0 (2026-10-05)

The first tagged release, with a Windows download (FeatherBench-v0.1.0-windows-x64.zip:
unzip, run FeatherBench.exe, no Python needed).

### FeatherBench, skips and the setup garage

- FeatherBench submissions by pull request: `featherbench/submissions/<github user>.json`
  (settings only, never code, known parts, at most 20,000 params). `rsi featherbench verify`
  checks a file exactly as CI does and runs the official benchmark; `rsi featherbench
  leaderboard` ranks CI's results into `featherbench/leaderboard.json`. CI: a read-only
  `pull_request` check that shows the score, and a main-only workflow that scores merged
  submissions on Linux and commits the leaderboard. House entry: `tactical-drone` (10 / 12
  solved at 346 params on CI's Linux runner). Rules in `featherbench/README.md`; the BridgeBench prompt in
  `docs/featherbench-prompt.md`; CLAUDE.md and AGENTS.md describe the attempt loop.
- UI: the setup garage (ghost lap, setup sheet, 5 laps, ask the machine) and a FeatherBench
  menu (run on this setup, top 10 with identicons, how to attempt).

- custom nn (Wide) now really uses skip connections, one per stage like the activations:
  `skip` wraps perc, new `skip_decide`, `skip_relate`, `skip_prepare` (default `same`) wrap
  decide, relate and prepare. Before, Wide stored `skip` but never applied it. With every
  skip `none` the network and its numbers are bit-identical (golden 99.6 / 97.5 unchanged).
  `concat` widens the next layer; a stage on `same` falls back to none where the main skip
  can't span a width change; an explicit misfit is refused at build time with a readable error.
- UI: Skip is enabled for custom nn, with skip: decide / relate / prepare dropdowns.
- New default: a residual skip around decide (`skip_decide = "residual (same width)"`), same
  341 params. Over seeds 0-4 at 7000 steps test accuracy goes from 80.3 % mean / 36.7 % min
  to 95.2 % / 93.3 % (2000 fresh points: 84.0 / 40.7 to 96.1 / 93.7). Seed 0 on V's Windows
  box: 100 % train / 95.8 % test (was 99.6 / 97.5 on its 120 test points, 96.7 → 97.6 fresh).
  The old recipe stays available as "fourier-gauss-v0" (`skip_decide = "none"`): a named
  golden, the doctor fingerprint, and a UI recipe. Room above it: concat skips reach 98.8 %
  at 546 params.
- The benchmark is named **FeatherBench**: fewest params that solves every pattern wins.
  Ids are now `featherbench-general-v1`, `featherbench-classic-v1`, `featherbench-quick-v1`;
  `general-v1`, `classic-v1`, `quick-v1` stay accepted (CLI, `bench:<id>` objectives, and
  `bench rank` of old submissions), with unchanged definition hashes.
  `python -m rsi featherbench` is an alias of `bench`. Schema ids are unchanged.

### Console, core and UI

Everything below also ships in 0.1.0 (it was prepared as "1.0.0"). The default config's numbers are unchanged: the new `Session`
is checked bit-for-bit against the old one (`tests/test_core_equivalence.py`). The
headline 99.6 / 97.5 (default, 7000 steps) was measured on Windows 11; other platforms
and torch versions give other numbers (Linux, torch 2.14: 98.3 / 90.0).

### Core (`nncore/`)

- `Session.configure` is transactional: if a change fails, data, model and optimizer stay
  exactly as they were. It also validates part names, value types and ranges first, with
  readable errors and did-you-mean suggestions.
- Each session has its own torch RNG stream, so other sessions, failed configures and
  your own `torch.rand` calls no longer shift its numbers.
- The default model loads without `import my_parts`: registries load `my_parts` lazily on
  a miss (`nncore.load_plugins`, env `NNCORE_PLUGINS`). `Registry.get(name, default)`
  keeps dict semantics.
- `Session(cfg)` copies its config instead of sharing yours. New opt-ins:
  `configure(skip_irrelevant=, keep_optimizer_state=)`, `replace_config(cfg)`,
  `train(stop_on_nonfinite=)` with `diverged_at`, `evaluate_fresh(n)` (accuracy on new
  points), `evaluate(extra_metrics=)`. `evaluate`/`predict` run in eval mode;
  `predict` takes numpy arrays.
- Readable errors for zero points, a single class, or an empty train split.
- New modules: `configio` (config <-> JSON, settings files shared with the UI, name
  aliases like `x^2`, bounds, hashing), `schema` (which model reads which field, param
  counts without touching the RNG, the `describe` catalog), `run` (`run_unit`: one
  JSON-safe training cell, NaN -> null).
- `nncore.__version__ = "1.0.0"`; torch threads come from `NNCORE_THREADS` (default 1).

### Parts

- Activations: `snake`, `cos`, `selu`, `softsign`, `hardtanh`, `prelu`, `x·sin x`.
- Features: `cos x`, `cos y`, `θ (atan2)`.
- Initializers: `xavier uniform (keep expansion)` and `kaiming normal (keep expansion)`
  (leave the Fourier layer alone), `orthogonal`, `xavier normal`, `lecun normal`. The
  original `xavier uniform` and `kaiming normal` are unchanged.
- Optimizers `NAdam`, `RAdam`, `Adamax`, `SGD+nesterov`; loss `focal (γ=2)`; schedules
  `cosine (over extra.total_steps)`, `cosine warm restarts (T0=1000)`,
  `warmup 200 + constant`; train steps `skip non-finite`, `clip grad norm (extra)`.
- Extra metrics (`bal_acc`, `macro_f1`, `worst_class_acc`, `mean_conf`, `ece`) in their own
  registry, so the existing output is unchanged.
- mlp: a skip that can't combine widths (`add`, `concat`) now fails at build time with a
  readable message instead of at the first step; `concat` is sized correctly.
- `parse_layers` errors name the bad token; `format_layers` is its inverse.
- `make_expansion(name, n_in, scale=None, **knobs)` passes knobs to `**kw` factories.
- `Spiral (k arms)` with fewer than k points and `test_frac` outside [0, 1) raise readable
  errors. New parts are appended, so dropdown order is unchanged.

### Datasets

- A zoo of 25 new datasets (rings, bullseye, pinwheel, Voronoi, hex tiling, Mandelbrot,
  yin-yang, imbalanced and label-noise sets, ...).
- Suites: `classic`, `zoo`, `general`, `stretch`, `sanity`, `all` (`--suite NAME`).
- Families with a size knob: `spiral[k]`, `checkerboard[c]`, `rings[k]`, `stripes[k]`,
  `blobs[k]`. Use them as a dataset name anywhere; they stay out of the UI dropdown.

### rsi console (new)

`python -m rsi <command>` (same as `python headless.py <command>`): one JSON document on
stdout per command, stable exit codes, errors as JSON instead of tracebacks. See AGENTS.md.

- `describe`, `schema`, `check`: levers, parts, which model reads what; validate a config
  and count its params without training.
- `run`: one config over seeds x datasets, in parallel worker processes, with `--map` /
  `--png` decision-boundary output and settings files the UI can load.
- `sweep` (grid, one-at-a-time, random) and `evolve` (resumable evolutionary search with
  holdout seeds), both with `--background`, `status`, `wait`, `stop`, `leaderboard`.
- `bench`: the fewest-params benchmark. Frozen, versioned definitions (`general-v1`,
  `classic-v1`, `quick-v1`), submission files, `bench rank` leaderboards, and an
  `evolve --objective bench:<id>` search for the smallest net that solves everything.
- `complexity`: steps a model's capacity up a ladder per family size and fits
  `params ~ c * k^p`, an empirically discovered complexity exponent.
- `export` / `open`: a winner as a settings file, opened in the window
  (`nn_playground.py --load FILE`).
- `runs list|query|show|stats` over a sqlite store of every trial; finished cells are
  cached by config and code fingerprint, so repeats are free. `replay` re-runs a trial and
  checks it is bit-identical; `repl` drives one live session over JSONL; `doctor` checks
  the environment and golden numbers.
- `agent_parts/`: where AI agents put their own parts (`--parts agent_parts.x`).

### headless.py

- Same output format as before. Fixed: `--report-every 0` no longer loops forever,
  every `--compare` value is checked before anything runs, output is line-buffered when
  piped, kebab-case flags (`--n-points`), exit codes 2 (usage) / 3 (bad config), and a
  warning when you set a field the model doesn't read (for example `--layers` on the
  default model).

### UI

- see UI notes

### Docs

- AGENTS.md (for AI agents), CONTRIBUTING.md (with the licence grant), this changelog,
  a PR template, a wider `.gitignore` (includes `runs/`).

### Compatibility notes

- Validation is stricter: unknown part names now fail on `Session(cfg)` even for fields
  the model doesn't read.
- Changing `my_parts.py` (or any `nncore` file) changes the code fingerprint, so the rsi
  cache recomputes those cells once. That is on purpose.
