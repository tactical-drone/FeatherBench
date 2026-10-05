<!--
Part of nn-playground. AGPL-3.0; for other licensing see COMMERCIAL.md.

README section for the rsi console, ready to paste. Whoever owns README.md:
- paste everything from the "## Driving it from scripts and AI agents" heading to the end
  of this file, after "Using the core from code" and before "## Controls";
- add one line to the Layout block:  rsi/  agent_parts/  AGENTS.md   the rsi console (scripts, agents)
- leave the Robots section byte-identical.
Also worth fixing while you are in there (from the code review, optional):
- "Using the core from code" sets layers="4:sin", which the default model ignores: add model="mlp".
- the Challenges and the "Hidden layers" control describe mlp knobs: say "with --model mlp".
-->

## Driving it from scripts and AI agents (rsi console)

The rsi console is the same nets without the window, built for scripts and AI agents:
every command prints exactly one JSON document, errors come back as JSON with a code and
a did-you-mean, and the exit code says what happened. Nothing extra to install; use the
same venv as the window (Python 3.10+). `python headless.py <command>` is the same as
`python -m rsi <command>`, and the old `headless.py` flags still work as before.

    python -m rsi describe                                   # start here: every lever, part, suite
    python -m rsi check --model mlp --layers 8:sin,8         # validate + param count, no training
    python -m rsi run --model mlp --layers 8:sin,8 --steps 3000 --seeds 0-4 --workers 4
    python -m rsi run --dataset Moons --steps 2000 --map 48x24      # the boundary as ASCII art
    python -m rsi sweep --vary activation=tanh,relu,gelu --model mlp --seeds 0-2 --name act
    python -m rsi evolve --set dataset="Spiral (5 arms)" --generations 20 --background --name s5a
    python -m rsi wait s5a --timeout 540
    python -m rsi leaderboard s5a --top 5
    python -m rsi export s5a --rank 1 -o winner.json --open     # load the winner in the window
    python -m rsi bench --benchmark quick-v1 --model mlp --layers 8,8

| Command | What it does |
|---|---|
| `describe`, `schema` | levers, parts, suites, which model reads which field; JSON Schemas of the outputs |
| `check` | resolve and validate a config, count its params, no training |
| `run` | one config over seeds x datasets, in parallel; `--map`, `--png`, `--save-settings` |
| `sweep` | grid (`--vary`), one-at-a-time (`--oat`) or random configs, ranked |
| `evolve` | evolutionary search with holdout seeds; resumable |
| `bench` | the fewest-params benchmark: `bench list`, `bench --benchmark ID --submit me.json`, `bench rank FILES` |
| `complexity` | discovered complexity exponent over a dataset family |
| `status`, `wait`, `stop` | follow a `--background` run |
| `leaderboard`, `export`, `open` | best entries; a winner as a settings file; open it in the window |
| `runs list\|query\|show\|stats` | every trial ever run, queryable (`--where "test_acc>=0.95"`) |
| `replay` | re-run a trial and check the numbers are bit-identical |
| `repl` | one live session over JSONL: train, tweak a knob, train on |
| `doctor` | environment checks, golden numbers, `--calibrate` ms/step |

Results land in `runs/` (git-ignored). Every finished cell is cached by config and code
fingerprint, so asking twice is free; editing `my_parts.py` or `nncore/` changes the
fingerprint, so old results never mix with new code.

**The fewest-params benchmark.** Whoever solves a benchmark with the fewest parameters
wins, since it is all about memory and training speed. A benchmark is a frozen dataset list
plus a fixed budget: `general-v1` is the 12 `general` datasets, 600 points, noise 0.05,
seeds 0-2, 3000 steps. A dataset counts as solved when the mean accuracy over the three
seeds on 2000 fresh points is at least 90 %. Your score is `params_max`, your net's param
count on its biggest dataset: "a net of at most P params solves everything". `bench rank`
sorts submissions by solved-everything, then fewest params, then mean accuracy. Published
benchmarks never change (a fix becomes `general-v2`), and a submission records the
benchmark hash, the full config, the platform and the code fingerprint, so anyone can re-run it.
`evolve --objective bench:general-v1` searches for the smallest solver for you.

**The complexity exponent.** Big O tells you how cost should grow; `complexity` measures
it. Pick a family with a size knob (`spiral` arms, `checkerboard` cells, `rings`,
`stripes`, `blobs`) and a recipe. For each size it widens the net one rung at a time
(width for the custom nn, every hidden layer for mlp) until the mean fresh accuracy hits
the threshold (0.95), keeps the smallest net that made it, then fits a line through
log(params) against log(size). The slope is the exponent you discovered:

    python -m rsi complexity --family spiral --sizes 2-6 --model mlp --layers 8,8
    ... params ~ 12.3 * k^1.42 (r^2 0.97): empirical complexity exponent 1.42   (made-up numbers)

Exponent 1 means each extra arm costs the same; above 1, every arm costs more than the
last. Same family, two recipes: the lower exponent has the better inductive bias.
`spiral[7]` style names work as a dataset anywhere.

The full guide for agents (golden rules, budgets, recipes, error codes) is
[AGENTS.md](AGENTS.md).
