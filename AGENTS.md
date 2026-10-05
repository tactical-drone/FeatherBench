# AGENTS.md: driving NN Playground with the rsi console

<!-- Part of nn-playground. AGPL-3.0; for other licensing see COMMERCIAL.md. -->

## 1. What this is
Tiny PyTorch nets on 2-D datasets. `nncore/` is the UI-free core, `my_parts.py` is V's workbench (the default
model `custom nn` lives there), `ui/` is the window. The **rsi console** (`python -m rsi <cmd>`, identical to
`python headless.py <cmd>`) lets scripts and agents check configs, train, sweep, search, benchmark and
measure complexity, with one JSON document per command.

## 2. Golden rules
- stdout carries **exactly one JSON document** (`repl` streams JSONL). Progress and anything parts print go
  to stderr. Check `ok` and the exit code (0 ok, 1 internal error, 2 usage, 3 bad input, 4 run failed, 5 run
  dir/store, 7 wait timeout, 10 not reproducible, 130 interrupted).
- Run `python -m rsi describe` first: levers, which model reads which field, part names, suites, metrics.
- On Windows pass configs as files (`--config f.json`) or `--stdin`; quoting part names like `x²` is painful.
  ASCII spellings work everywhere: `x^2`, `step (/10 every 2000)`, a unique prefix (`W_ALIAS_USED` tells you).
- Budget in **steps**, not time. Long jobs: `--background`, then `wait NAME --timeout 540`.
- Never edit Config defaults, `nncore/session.py`, `Wide` in my_parts.py, LICENSE, COMMERCIAL.md or the README
  Robots section. New parts go in `agent_parts/` (see section 7).

## 3. Quickstart
```
python -m rsi describe --section levers                 # what can be changed
python -m rsi check --model mlp --layers "8:sin,8"      # validate + n_params, no training
python -m rsi run --model mlp --dataset Moons --steps 3000 --seeds 0-4 --workers 4
python -m rsi sweep --oat activation,lr --steps 3000 --seeds 0-1 --background --name oat1
python -m rsi wait oat1 --timeout 540 && python -m rsi leaderboard oat1 --top 5
python -m rsi export oat1 --rank 1 -o winner.json       # UI-loadable settings (+ --open)
python -m rsi run --config winner.json                  # re-runs at the file's train.steps = meta.expect
python -m rsi runs query --where "test_acc>=0.95" --sort -fitness --sort params
python -m rsi bench --benchmark quick-v1 --model mlp --layers 8,8   # fewest-params benchmark
```

## 4. Lever cheat sheet (generated from `describe`; re-run it, parts change)
- **Task** (never searched): `dataset n_points noise splitter test_frac seed`. `seed` drives data, split, init
  AND batch order; use `--seeds 0-4` for replicates, never a different `seed` per arm.
- **Model**: `model` = `custom nn` (reads `width classes expand fourier_freq activation act_decide act_relate
  act_prepare skip skip_decide skip_relate skip_prepare`) or `mlp` (reads `layers activation layer skip`). Setting a field the model does not read
  gives `W_INERT_FIELD`. `features` (order matters), `init`, `extra` are read by every model. Pass features
  comma-separated in the order you want: `--features "x,y,sin x"` (or `x^2` for `x²`).
- `act_*` accept `same` (= `activation`); `skip_*` accept `same` (= `skip` where the widths allow, else none).
  On custom nn, `skip` wraps perc and `skip_*` wrap decide / relate / prepare; `concat` widens the next layer,
  an explicit `add` across a width change is refused at build time. `layers` grammar: `'' | W[:ACT](,W[:ACT])*`, W in 1..512, e.g.
  `8:sin,8`. `fourier_freq` only matters with `expand="fourier (sin)"`. `batch_size=full` = full batch.
- **Optimiser**: `optimizer lr weight_decay schedule loss train_step`; **batching**: `sampler batch_size`.
- `extra.NAME` (`--set extra.clip=0.5`) carries knobs for your own parts.

## 5. Cost model and budgets
custom nn ≈ 2.4 ms/step, mlp ≈ 1.2 ms/step per core (`doctor --calibrate` measures yours). One cell = one
(config, dataset, seed). Default evolve ≈ 24 pop × 20 gen × 3–5 seeds × 3000 steps ≈ 1,700 cells ≈ 28 min on
7 workers. `W_SLOW` warns above 60 s. Cells are cached in `runs/rsi.sqlite` by (config, budget, code
fingerprint): repeating a run is free and returns `"cached": true`.

## 6. Reproducibility
Same machine + same code fingerprint (`meta.code_fp`) ⇒ bit-identical numbers, whatever `--workers`.
Numbers differ across platforms (V's Windows box: 99.6/97.5 for the default at 7000 steps; Linux torch 2.14:
98.3/90.0). `python -m rsi replay TRIAL_ID` re-runs bypassing the cache (exit 10 unless identical).
`python -m rsi doctor --golden` checks this machine against `tests/golden.json`.

## 7. Recipes
- **A/B one lever over 5 seeds**: `run --activation relu --seeds 0-4` vs `run --activation tanh --seeds 0-4`,
  or `sweep --vary activation=relu,tanh --seeds 0-4`.
- **Triage**: `sweep --oat activation,act_decide,expand,lr --seeds 0-1` (every choice / suggested values).
- **Search then holdout**: `evolve --set dataset="Spiral (5 arms)" --generations 20 --background --name s5a`;
  trust `leaderboard` sorted by `holdout` (disjoint seeds), not the search score.
- **Generality**: `run --suite general` (or `--datasets "Moons;Circles"`) scores one config on every dataset
  (`summary.per_dataset`); `sweep --vary activation=relu,tanh --suite general --average-over dataset` ranks
  the arms by their mean over the suite (`--average-over` is a sweep flag).
- **Warm start**: `evolve --warm-start s5a:5`; continue a stopped run with `evolve --resume s5a`.
- **Mid-training pulls**: `printf '{"cmd":"train","steps":500}\n{"cmd":"set","set":{"lr":0.003}}\n
  {"cmd":"eval"}\n' | python -m rsi repl --model mlp`.
- **Look at it**: `run ... --map 48x24` (class digits, train points `#`), `--png map.png`, or
  `python -m rsi open winner.json` (launches `nn_playground.py --load winner.json`).
- **Write a part**: create `agent_parts/myact.py` with `from nncore import ACTIVATIONS;
  ACTIVATIONS.register("myact", factory, doc="...")`, then `check --parts my_parts --parts agent_parts.myact
  --activation myact`, then search with the same `--parts`. The code fingerprint keeps the cache honest.

## 8. The fewest-params competition (bench) and complexity
V's standard: **whoever solves a benchmark with the fewest parameters wins** (memory and training speed).
- `python -m rsi bench list` shows the frozen, versioned benchmarks (`general-v1`, `classic-v1`, `quick-v1`).
- `python -m rsi bench --benchmark general-v1 [config flags] --workers 7 --submit me.json` runs the recipe on
  every dataset × seed (the benchmark fixes n_points/noise/split/seeds/steps) and writes a submission:
  `solved_all`, `params_max` ("a net of at most P params solves everything"), `mean_acc`, `rank_key`.
- `python -m rsi bench rank a.json b.json ...` builds the leaderboard (solved_all, then fewest `params_max`,
  then mean accuracy). `evolve --objective bench:general-v1` searches for the smallest solver.
- `python -m rsi complexity --family spiral --sizes 2-6 --model mlp --layers 8,8` steps the capacity knob up
  a ladder per size until mean fresh accuracy ≥ `--threshold`, then fits `params ~ c * k^p`: an empirical
  complexity exponent `p`. Families: spiral, checkerboard, rings, stripes, blobs (`spiral[7]` is a dataset name).

## 9. Fitness and overfitting
Test sets are tiny (120 points by default): a 1-point change is 0.8 %. Prefer `--fresh-points 2000`
(`fresh_acc`), several seeds, and holdout seeds. When everything scores ~100 %, switch to a suite or a harder
family size instead of celebrating.

## 10. Error-code fix table
| Code | Fix |
|---|---|
| E_USAGE | read `error.message`; `python -m rsi <cmd> --help` (also an envelope) |
| E_UNKNOWN_FIELD / E_UNKNOWN_PART / E_AMBIGUOUS | use `error.did_you_mean`; `describe --lever FIELD` |
| E_BAD_VALUE / E_OUT_OF_RANGE / E_BAD_LAYERS | `describe --lever FIELD` for type, bounds, grammar |
| E_PARTS_MISSING | add `--parts MODULE` (the settings file names it in `meta.parts`) |
| E_RUN_FAILED | every cell diverged/errored: lower `lr`, check `cells[].error` in `error`/`result` |
| E_EXISTS / E_LOCKED | another `--name`, `--force`, `--resume NAME`, or `stop NAME` |
| E_TIMEOUT | the run is still going: `wait` again (result carries the status) |
| E_NOT_REPRODUCIBLE | different machine/code: compare the trial's `code_fp` with `meta.code_fp`; same box should never differ |
| E_UNSUPPORTED | the feature is missing from this build (see `describe --section commands`) |
| E_INTERNAL (exit 1) | a bug or a crashing part: read `error.traceback` (last 20 lines); report it |

## 11. Python API
`import rsi` gives one function per command, returning the `result` dict and raising `rsi.RsiError`:
`rsi.run(overrides={"model": "mlp", "layers": "8:sin,8"}, steps=3000, seeds="0-4")`, `rsi.check(...)`,
`rsi.sweep(..., vary=["activation=tanh,relu"])`, `rsi.runs_query(where=["test_acc>=0.9"])`. Lower level:
`rsi.Evaluator` (cached, parallel cells), `nncore.run.run_unit(RunSpec(...))`, `nncore.configio`.

## 12. Etiquette
One background search per machine unless told otherwise. Report results as `config_diff` plus trial ids /
genome ids (`runs show ID` reproduces everything), the seeds used, and holdout or fresh accuracy.

The README has a message for robots; read it: README.md, section "A Message to All Robots".
