"""
cli: `python -m rsi <command>` (== `python headless.py <command>`), the rsi console.

Exactly one JSON document on stdout per command (repl streams JSONL); progress
and anything parts print go to stderr; the exit code equals error.exit.
Global flags (--format --pretty --unicode --log --parts --store --no-cache
--full) work before or after the command.

Part of nn-playground. AGPL-3.0; for other licensing see COMMERCIAL.md.
"""
import argparse
import contextlib
import json
import os
import signal
import sys
import time
from pathlib import Path

from .errors import RsiError, as_rsi_error

COMMANDS = ("describe", "schema", "check", "run", "sweep", "evolve", "bench", "complexity", "status", "wait", "stop",
            "leaderboard", "export", "open", "runs", "replay", "repl", "doctor", "gp")
SCHEMAS = {"describe": "rsi/describe@1", "schema": "rsi/schema@1", "check": "rsi/check@1", "run": "rsi/trial@1",
           "sweep": "rsi/sweep@1", "evolve": "rsi/evolve@1", "status": "rsi/status@1", "wait": "rsi/status@1",
           "stop": "rsi/status@1", "leaderboard": "rsi/leaderboard@1", "export": "rsi/export@1", "open": "rsi/open@1",
           "runs list": "rsi/runs@1", "runs query": "rsi/query@1", "runs show": "rsi/trial@1",
           "runs stats": "rsi/stats@1", "replay": "rsi/replay@1", "doctor": "rsi/doctor@1", "gp check": "rsi/gp-check@1",
           "bench": "rsi/bench@1", "bench list": "rsi/bench-list@1", "bench rank": "rsi/bench-leaderboard@1",
           "complexity": "rsi/complexity@1"}
GLOBAL_VALUE = ("--format", "--log", "--parts", "--store")
GLOBAL_FLAG = ("--pretty", "--unicode", "--no-cache", "--full")


class _Help(Exception):
    def __init__(self, text):
        super().__init__("help")
        self.text = text


class Parser(argparse.ArgumentParser):
    """argparse that raises E_USAGE (so usage errors also produce the envelope), with
    did_you_mean for mistyped commands, choices and flags."""
    def __init__(self, *a, **kw):
        kw.setdefault("allow_abbrev", False)
        super().__init__(*a, **kw)

    def error(self, message):
        raise RsiError(f"{self.prog}: {message}", "E_USAGE", hint=f"{self.prog} --help")

    def _check_value(self, action, value):
        if action.choices is not None and value not in action.choices:
            import difflib
            names = [str(c) for c in action.choices]
            what = "command" if isinstance(action, argparse._SubParsersAction) else \
                "value for " + ("/".join(action.option_strings) or action.dest)
            raise RsiError(f"{self.prog}: unknown {what} {value!r}", "E_USAGE", value=value, allowed=names,
                           did_you_mean=difflib.get_close_matches(str(value), names, n=3, cutoff=0.5),
                           hint=f"{self.prog} --help")
        return super()._check_value(action, value)

    def parse_args(self, args=None, namespace=None):
        ns, extras = self.parse_known_args(args, namespace)
        if extras:
            raise _unknown_args(self, ns, extras)
        return ns

    def print_help(self, file=None):
        raise _Help(self.format_help())

    def exit(self, status=0, message=None):
        if status:
            raise RsiError(message or "usage error", "E_USAGE")
        raise _Help(message or "")


def _unknown_args(root, ns, extras):
    """E_USAGE for unrecognized arguments, named by the (sub)command, with close flag names."""
    import difflib
    p, names = root, []
    for dest in ("command", "runs_cmd", "gp_cmd"):
        sub = getattr(ns, dest, None)
        spa = next((a for a in p._actions if isinstance(a, argparse._SubParsersAction)), None)
        if sub is None or spa is None or sub not in spa.choices:
            break
        p, names = spa.choices[sub], names + [sub]
    flags = sorted(set(p._option_string_actions) | set(GLOBAL_FLAG) | set(GLOBAL_VALUE))
    mean = []
    for x in extras:
        if x.startswith("--"):
            mean += difflib.get_close_matches(x.partition("=")[0], flags, n=3, cutoff=0.75)
    prog = " ".join(["python -m rsi"] + names)
    return RsiError(f"{prog}: unrecognized arguments: {' '.join(extras)}", "E_USAGE", value=extras[0],
                    did_you_mean=list(dict.fromkeys(mean))[:3], hint=f"{prog} --help")


# ---------------------------------------------------------------- globals
def split_globals(argv):
    """Pull the global flags out of argv wherever they are; returns (globals, rest)."""
    g = {"format": "json", "pretty": False, "unicode": False, "log": "text", "parts": [], "store": None,
         "no_cache": False, "full": False}
    rest, i = [], 0
    while i < len(argv):
        a = argv[i]
        if a == "--":
            rest += argv[i:]
            break
        head, eq, val = a.partition("=")
        if head in GLOBAL_FLAG and not eq:
            g[head[2:].replace("-", "_")] = True
        elif head in GLOBAL_VALUE:
            if not eq:
                if i + 1 >= len(argv):
                    raise RsiError(f"{head} needs a value", "E_USAGE")
                val = argv[i + 1]
                i += 1
            key = head[2:]
            if key == "parts":
                g["parts"].append(val)
            else:
                g[key] = val
        else:
            rest.append(a)
        i += 1
    if g["format"] not in ("json", "text"):
        raise RsiError(f"--format {g['format']!r}: use json or text", "E_USAGE")
    if g["log"] not in ("text", "jsonl", "quiet"):
        raise RsiError(f"--log {g['log']!r}: use text, jsonl or quiet", "E_USAGE")
    return g, rest


# ---------------------------------------------------------------- flag groups
def add_config_flags(p, skip=()):
    from nncore.configio import FIELDS, FIELD_HELP
    g = p.add_argument_group("config (precedence: Config() < --config < --from < --stdin < --FIELD < --set-json < --set)")
    g.add_argument("--config", help="settings doc, bare config dict, or trial record (JSON file)")
    g.add_argument("--from", dest="from_ref", help="trial id (prefix) from the store, or RUNNAME[:rank]")
    g.add_argument("--stdin", action="store_true", help="read a JSON config object from stdin")
    for f in FIELDS:
        if f == "extra" or f in skip:
            continue
        names = [f"--{f}"] + ([f"--{f.replace('_', '-')}"] if "_" in f else [])
        g.add_argument(*names, dest=f"f_{f}", metavar="V", help=FIELD_HELP.get(f, ""))
    g.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="field or extra.NAME (repeatable)")
    g.add_argument("--set-json", action="append", default=[], metavar="JSON", help="object of overrides")


def add_budget_flags(p, steps=2000, workers=1, steps_help=None):
    g = p.add_argument_group("budget and replicates")
    g.add_argument("--steps", type=int, default=steps,
                   help=f"training steps per cell (default {steps_help or steps})")
    g.add_argument("--eval-every", type=int, default=None, help="curve granularity; never changes the numbers")
    g.add_argument("--seeds", help="0,1,2 or 0-4 or 0-4,10 (default: the config's seed)")
    g.add_argument("--n-seeds", type=int, help="seeds seed..seed+K-1")
    g.add_argument("--datasets", help="';'-separated dataset names")
    g.add_argument("--suite", help="classic | general | zoo | all | ...")
    g.add_argument("--max-seconds", type=float, help="per-cell time limit (status timeout, never cached)")
    g.add_argument("--es-step", type=int, help="early stop: at this step ...")
    g.add_argument("--es-min-acc", type=float, help="... stop if test acc < this")
    g.add_argument("--fresh-points", type=int, default=0, help="also score N fresh points (fresh_*)")
    g.add_argument("--metrics", default="", help="extra metrics, e.g. bal_acc,macro_f1")
    g.add_argument("--workers", type=int, default=workers,
                   help="spawned processes; 0/1 = in-process (default " +
                   (f"{workers})" if workers is not None else "cpu count - 1, at most 8)"))
    g.add_argument("--tag", action="append", default=[], help="stored with the trial (repeatable)")


def build_parser():
    root = Parser(prog="python -m rsi", description="rsi console: drive NN Playground from scripts and agents. "
                  "Start with: python -m rsi describe", formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = root.add_subparsers(dest="command", metavar="COMMAND", parser_class=Parser)

    p = sub.add_parser("describe", help="levers, models, parts, suites, metrics, objectives, commands, errors")
    p.add_argument("--section", choices=["levers", "models", "parts", "suites", "metrics", "objectives", "commands",
                                         "errors", "env", "benchmarks", "families"])
    p.add_argument("--lever")
    p.add_argument("--model")

    p = sub.add_parser("schema", help="JSON Schema of a document")
    p.add_argument("name", nargs="?")

    p = sub.add_parser("check", help="resolve, validate and build a config")
    add_config_flags(p)
    p.add_argument("--save", metavar="FILE")
    p.add_argument("--diff", action="store_true", help="omit the full config")

    p = sub.add_parser("run", help="train one config over seeds x datasets")
    add_config_flags(p)
    add_budget_flags(p, steps=None, steps_help="the --config file's train.steps, else 2000")
    p.add_argument("--map", nargs="?", const="48x24", metavar="WxH")
    p.add_argument("--png", metavar="FILE")
    p.add_argument("--save-settings", metavar="FILE")
    p.add_argument("--curve", action="store_true")

    p = sub.add_parser("sweep", help="grid / one-at-a-time / random over configs")
    add_config_flags(p)
    add_budget_flags(p, workers=None)
    p.add_argument("--vary", action="append", default=[], metavar="FIELD=v1,v2",
                   help="grid axis (repeatable); ';' splits values that hold commas; @all, @all+same, "
                        "@suite:NAME (dataset)")
    p.add_argument("--vary-json", metavar="JSON", help='grid axes as an object, e.g. {"layers": ["8,8", "16"]}')
    p.add_argument("--oat", metavar="FIELD[,FIELD]",
                   help="one-at-a-time around the base: every choice, or the lever's suggested values")
    p.add_argument("--random", type=int, metavar="N", help="N random configs from --space")
    p.add_argument("--space", help="space file for --random (default: rsi/spaces/default.json)")
    p.add_argument("--sample-seed", type=int, default=0, help="rng seed for --random (default 0)")
    p.add_argument("--rank-by", default="test_acc",
                   help="test_acc | train_acc | fresh_acc | test_loss | params, optionally :min/:median "
                        "(default test_acc)")
    p.add_argument("--average-over", action="append", default=[], metavar="FIELD",
                   help="group results averaged over this field, e.g. dataset (repeatable)")
    p.add_argument("--max-runs", type=int, default=500, help="refuse grids larger than this (default 500)")
    p.add_argument("--top", type=int, default=20, help="trials listed in the result (default 20)")
    _run_dir_flags(p)

    p = sub.add_parser("evolve", help="evolutionary search (resumable)")
    add_config_flags(p, skip=("init",))  # --init is the population init here (see EVOLVE_FLAGS)
    add_budget_flags(p, steps=None, workers=None, steps_help=3000)
    for flag, kw in EVOLVE_FLAGS:
        p.add_argument(flag, **kw)
    _run_dir_flags(p)
    p.add_argument("--resume", metavar="NAME", help="continue a checkpointed search (only budgets may change)")
    p.add_argument("--allow-code-change", action="store_true",
                   help="with --resume: accept edited code (the elites are re-evaluated)")

    p = sub.add_parser("bench", help="fewest-params benchmark: bench [list | rank FILE... | --benchmark ID]")
    p.add_argument("action", nargs="?", help="list | rank (omit to run a benchmark)")
    p.add_argument("files", nargs="*")
    add_config_flags(p)
    p.add_argument("--benchmark", default="general-v1")
    p.add_argument("--workers", type=int)
    p.add_argument("--submit", metavar="FILE")
    p.add_argument("--steps", type=int, help="override the benchmark's steps (an unofficial result)")
    p.add_argument("--top", type=int)

    p = sub.add_parser("complexity", help="empirical complexity exponent over a dataset family")
    add_config_flags(p)
    p.add_argument("--family", required=True)
    p.add_argument("--sizes", required=True, help="2-6 or 2,3,5")
    p.add_argument("--threshold", type=float, default=0.95)
    p.add_argument("--ladder", help="capacity rungs, e.g. 1,2,3,4,6,8,12,16")
    p.add_argument("--knob", help="capacity field for models other than custom nn / mlp")
    p.add_argument("--seeds", default="0-2")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--workers", type=int)

    for name, hlp in (("status", "status of a run dir"), ("wait", "block until not running"),
                      ("stop", "ask a run to stop")):
        p = sub.add_parser(name, help=hlp)
        p.add_argument("target", metavar="NAME|DIR")
        if name == "status":
            p.add_argument("--history", type=int, default=10)
        if name == "wait":
            p.add_argument("--timeout", type=float, default=540.0)
        if name == "stop":
            p.add_argument("--now", action="store_true")
            p.add_argument("--wait", action="store_true")
            p.add_argument("--timeout", type=float, default=600.0)

    p = sub.add_parser("leaderboard", help="ranked entries of run dirs")
    p.add_argument("targets", nargs="+", metavar="NAME|DIR")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--sort", default="holdout", choices=["holdout", "score", "params", "acc"])
    p.add_argument("--pareto", action="store_true")

    p = sub.add_parser("export", help="settings file for an entry or trial")
    p.add_argument("target", metavar="NAME|DIR|TRIAL_ID")
    p.add_argument("--rank", type=int, default=1)
    p.add_argument("--id", dest="genome_id")
    p.add_argument("-o", "--out", metavar="FILE")
    p.add_argument("--steps", type=int)
    p.add_argument("--open", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="with --open: print the launch command only")

    p = sub.add_parser("open", help="open in the playground window (nn_playground.py --load FILE)")
    p.add_argument("target", metavar="FILE|NAME|TRIAL_ID")
    p.add_argument("--opengl", action="store_true")
    p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("runs", help="runs list | query | show | stats")
    rs = p.add_subparsers(dest="runs_cmd", metavar="list|query|show|stats", parser_class=Parser)
    q = rs.add_parser("list")
    q.add_argument("--kind", choices=["sweep", "evolve"])
    q.add_argument("--limit", type=int, default=20)
    q = rs.add_parser("query")
    q.add_argument("--where", action="append", default=[])
    q.add_argument("--sort", action="append", default=[])
    q.add_argument("--limit", type=int, default=20)
    q.add_argument("--fields")
    q.add_argument("--tag")
    q.add_argument("--run")
    q.add_argument("--objective")
    q = rs.add_parser("show")
    q.add_argument("id")
    q.add_argument("--curve", action="store_true")
    q = rs.add_parser("stats")
    q.add_argument("--group-by", required=True)
    q.add_argument("--where", action="append", default=[])
    q.add_argument("--metric", default="summary.test_acc.mean")

    p = sub.add_parser("replay", help="re-run a trial bypassing the cache")
    p.add_argument("trial_id")
    p.add_argument("--workers", type=int, default=1)

    p = sub.add_parser("repl", help="JSONL session on stdin / stdout")
    add_config_flags(p)

    p = sub.add_parser("doctor", help="environment checks")
    p.add_argument("--golden", action="store_true")
    p.add_argument("--record-golden", action="store_true")
    p.add_argument("--calibrate", action="store_true")

    p = sub.add_parser("gp", help="gp check EXPR")
    gs = p.add_subparsers(dest="gp_cmd", metavar="check", parser_class=Parser)
    q = gs.add_parser("check")
    q.add_argument("expr")
    return root


def _run_dir_flags(p):
    p.add_argument("--name", help="run dir name under the store (default: a timestamp)")
    p.add_argument("--out", metavar="DIR", help="run dir path instead of --name")
    p.add_argument("--force", action="store_true", help="replace an existing run dir")
    p.add_argument("--background", action="store_true",
                   help="detach and return at once; then status / wait / stop NAME")


def _f(**kw):
    return kw


EVOLVE_FLAGS = [  # section 6.9; None = the EvolveSettings default (named in the help)
    ("--space", _f(help="default | FILE: the genes to search (default rsi/spaces/default.json)")),
    ("--models", _f(help="restrict the model gene, e.g. \"custom nn\",mlp")),
    ("--genes", _f(help="search only these genes (comma list)")),
    ("--freeze", _f(help="keep these genes at the base value (comma list)")),
    ("--unfreeze", _f(help="search these although pinned or frozen by default (comma list)")),
    ("--pop", _f(type=int, help="population size (default 24)")),
    ("--generations", _f(type=int, help="generations (default 20)")),
    ("--elite", _f(type=int, help="best genomes copied unchanged each generation (default 2)")),
    ("--tournament", _f(type=int, help="tournament size for parent selection (default 3)")),
    ("--p-crossover", _f(type=float, help="crossover probability, else clone (default 0.7)")),
    ("--mutations", _f(type=float, help="mean mutation operations per child, >= 1 (default 1.5)")),
    ("--immigrants", _f(type=float, help="fraction of fresh random genomes per generation (default 0.1)")),
    ("--init", _f(help="population init: mixed | base | random (default mixed); any other value sets the "
                       "Config field init")),
    ("--seed-config", _f(action="append", help="settings file added to the first population (repeatable)")),
    ("--warm-start", _f(action="append", help="RUN[:K]: top K of an earlier run in the first population")),
    ("--max-seeds", _f(type=int, help="seeds a promising genome is escalated to (default 5)")),
    ("--escalate-top", _f(type=float, help="escalate genomes near the top fraction of pop (default 0.25)")),
    ("--escalate-margin", _f(type=float, help="score margin below that threshold (default 0.01)")),
    ("--holdout-seeds", _f(help="disjoint seeds that re-score the finalists (default 1000-1004)")),
    ("--holdout-top", _f(type=int, help="finalists re-scored on the holdout seeds; 0 = none (default 5)")),
    ("--holdout-every", _f(type=int, help="also run the holdout every K generations; 0 = only at the end "
                                          "(default 0)")),
    ("--metric", _f(choices=["test_acc", "train_acc", "fresh_acc", "test_loss"],
                    help="cell metric (default test_acc; fresh_acc adds --fresh-points 2000)")),
    ("--seed-agg", _f(choices=["mean", "median", "min", "cvar50"], help="over seeds (default mean)")),
    ("--reduce", _f(choices=["mean", "min"], help="over datasets (default mean)")),
    ("--std-weight", _f(type=float, help="score -= w * std over cells (default 0)")),
    ("--param-penalty", _f(type=float, help="score -= p * log2(params / param_ref) (default 0.005)")),
    ("--param-ref", _f(type=float, help="params with no penalty (default 64)")),
    ("--max-params", _f(type=int, help="reject bigger genomes without training (default 20000)")),
    ("--time-penalty", _f(type=float, help="score -= t * seconds per 1000 steps (default 0)")),
    ("--success-threshold", _f(type=float, help="accuracy counted as success in success_rate (default 0.9)")),
    ("--objective", _f(help="scalar | pareto | bench:<id> (default scalar; bench:<id> = fewest params that "
                            "solve the benchmark, see bench list)")),
    ("--fidelity", _f(help="S1,S2: train every genome S1 steps, the top third S2 (default off)")),
    ("--gp-activations", _f(action="store_true", default=None, help="also evolve gp: expression activations")),
    ("--gp-max-depth", _f(type=int, help="GP tree depth (default 4)")),
    ("--gp-max-nodes", _f(type=int, help="GP tree size (default 15)")),
    ("--gp-rate", _f(type=float, help="share of activation mutations that use GP (default 0.15)")),
    ("--gp-learnable-consts", _f(action="store_true", default=None, help="GP constants are trained (gpl:)")),
    ("--search-seed", _f(type=int, help="the GA's random seed (default 0)")),
    ("--time", _f(type=float, help="wall-clock budget in seconds (default none)")),
    ("--max-evals", _f(type=int, help="budget of newly trained cells (default none)")),
    ("--stall", _f(type=int, help="stop after G generations without improvement (default 8)")),
    ("--max-est-seconds", _f(type=float, help="reject genomes estimated slower than this per cell set "
                                              "(default 120)")),
    ("--cache-scope", _f(choices=["code", "config"], help="config: reuse cells across code edits (default code)")),
]


# ---------------------------------------------------------------- arg -> API
def config_sources(a, stdin):
    """(sources list, overrides dict) from the config flags, in precedence order."""
    from nncore.configio import FIELDS

    from . import api
    sources = []
    if getattr(a, "config", None):
        sources.append(a.config)
    if getattr(a, "from_ref", None):
        sources.append(api.config_from_ref(a.from_ref))
    if getattr(a, "stdin", False):
        try:
            doc = json.load(stdin)
        except ValueError as e:
            raise RsiError(f"--stdin: not valid JSON ({e})", "E_BAD_SETTINGS") from None
        if not isinstance(doc, dict):
            raise RsiError("--stdin: expected a JSON object", "E_BAD_SETTINGS")
        sources.append(doc)
    overrides = {}
    for f in FIELDS:
        v = getattr(a, f"f_{f}", None)
        if v is not None:
            overrides[f] = v
    for text in getattr(a, "set_json", []) or []:
        try:
            obj = json.loads(text)
        except ValueError as e:
            raise RsiError(f"--set-json: not valid JSON ({e})", "E_USAGE", value=text) from None
        if not isinstance(obj, dict):
            raise RsiError("--set-json takes a JSON object", "E_USAGE", value=text)
        overrides.update(obj)
    for item in getattr(a, "set", []) or []:
        k, eq, v = item.partition("=")
        if not eq or not k.strip():
            raise RsiError(f"bad --set {item!r}: expected KEY=VALUE", "E_USAGE", value=item)
        k = k.strip()
        overrides[k if k.startswith("extra.") else k.replace("-", "_")] = v
    return sources or None, overrides


def budget_kw(a):
    es = None
    if a.es_step is not None or a.es_min_acc is not None:
        if a.es_step is None or a.es_min_acc is None:
            raise RsiError("--es-step and --es-min-acc go together", "E_USAGE")
        es = (a.es_step, a.es_min_acc)
    kw = {"steps": a.steps, "seeds": a.seeds, "n_seeds": a.n_seeds, "datasets": a.datasets, "suite": a.suite,
          "max_seconds": a.max_seconds, "early_stop": es, "fresh_points": a.fresh_points,
          "metrics": [m for m in a.metrics.split(",") if m.strip()], "workers": a.workers, "tags": a.tag}
    if a.eval_every is not None:
        kw["eval_every"] = a.eval_every
    return kw


def _bg_path(a, g, command):
    from .store import default_root
    root = Path(g["store"]) if g["store"] else default_root()
    if getattr(a, "resume", None):
        return Path(a.out) if a.out else root / a.resume, None
    name = a.name or time.strftime(f"{command}-%Y%m%d-%H%M%S")
    return (Path(a.out) if a.out else root / name), (None if (a.name or a.out) else name)


def background(a, g, argv, command):
    """--background: check the run dir, then re-exec without --background (and --force)."""
    import shutil

    from . import rundir as rd
    if getattr(a, "stdin", False):
        raise RsiError("--stdin cannot be combined with --background; pass --config FILE", "E_USAGE")
    path, gen_name = _bg_path(a, g, command)
    if getattr(a, "resume", None):  # a live run (or one just launched) keeps its stdout.json / launch.json
        if not (path / "manifest.json").exists():
            raise RsiError(f"{path} is not a run dir", "E_NOT_FOUND", value=str(path))
        rd._check_lock(path)
        pid = (rd.read_json(path / "launch.json", {}) or {}).get("pid")
        if pid and pid != os.getpid() and rd.pid_alive(pid) and rd.read_status(path)["state"] in ("running", "stopping"):
            raise RsiError(f"{path} is still running (pid {pid})", "E_LOCKED", value=str(path),
                           hint=f"python -m rsi stop {path.name} --wait, then resume")
    elif (path / "manifest.json").exists():
        if not a.force:
            raise RsiError(f"run dir {path} exists", "E_EXISTS", value=str(path),
                           hint="pick another --name, pass --force, or --resume it")
        rd._check_lock(path)
        shutil.rmtree(path)
    child = rd.strip_flags(argv, flags_no_value=("--background", "--force"))
    if gen_name:
        child += ["--name", gen_name]
    return rd.launch_background(child, path, command)


def dispatch(a, g, argv, stdin, out):
    """Run the parsed command; returns (command label, schema id, result)."""
    from . import api
    cmd = a.command
    parts = g["parts"] or None
    cache = not g["no_cache"]
    log = None
    if g["log"] != "quiet":
        from .io import Logger
        log = Logger(g["log"])
    if cmd == "describe":
        return cmd, SCHEMAS[cmd], api.describe(section=a.section, lever=a.lever, model=a.model, parts=parts)
    if cmd == "schema":
        from . import schemas
        if not a.name:
            return cmd, SCHEMAS[cmd], {"names": list(schemas.NAMES)}
        doc = schemas.get(a.name)
        if doc is None:
            import difflib
            raise RsiError(f"unknown schema '{a.name}'", "E_USAGE", value=a.name, allowed=list(schemas.NAMES),
                           did_you_mean=difflib.get_close_matches(a.name, schemas.NAMES, n=3))
        return cmd, SCHEMAS[cmd], doc
    if cmd == "check":
        api.setup_parts(parts)
        src, ov = config_sources(a, stdin)
        return cmd, SCHEMAS[cmd], api.check(src, ov, save=a.save, diff=a.diff, parts=parts)
    if cmd == "run":
        api.setup_parts(parts)
        src, ov = config_sources(a, stdin)
        return cmd, SCHEMAS[cmd], api.run(src, ov, **budget_kw(a), map=a.map, png=a.png, save_settings=a.save_settings,
                                         curve=a.curve, cache=cache, parts=parts, on_event=log)
    if cmd in ("sweep", "evolve") and a.background:
        return cmd, "rsi/background@1", background(a, g, argv, cmd)
    if cmd == "sweep":
        api.setup_parts(parts)
        src, ov = config_sources(a, stdin)
        vj = None
        if a.vary_json:
            try:
                vj = json.loads(a.vary_json)
                assert isinstance(vj, dict)
            except (ValueError, AssertionError):
                raise RsiError("--vary-json takes an object {field: [values]}", "E_USAGE") from None
        res = api.sweep(src, ov, vary=a.vary or None, vary_json=vj, oat=a.oat, random_n=a.random, space=a.space,
                        sample_seed=a.sample_seed, rank_by=a.rank_by, average_over=a.average_over,
                        max_runs=a.max_runs, top=a.top, name=a.name, out=a.out, force=a.force, cache=cache,
                        parts=parts, on_event=log, **budget_kw(a))
        return cmd, SCHEMAS[cmd], res
    if cmd == "evolve":
        api.setup_parts(parts)
        src, ov = config_sources(a, stdin)
        kw = {k: v for k, v in budget_kw(a).items() if v is not None and v != []}
        kw.pop("workers", None)
        if kw.get("seeds") is not None:
            kw["seeds"] = api.parse_seeds(kw["seeds"])
        if a.init is not None and a.init not in ("mixed", "base", "random"):
            ov["init"], a.init = a.init, None  # a weight-init part name: the Config field
        for flag, _ in EVOLVE_FLAGS:
            k = flag[2:].replace("-", "_")
            v = getattr(a, k, None)
            if v is None or k == "space":
                continue
            if k == "holdout_seeds":
                v = api.parse_seeds(v, "--holdout-seeds")
            elif k in ("models", "genes", "freeze", "unfreeze"):
                v = [x.strip() for x in v.split(",") if x.strip()]
            elif k == "fidelity":
                try:
                    v = [int(x) for x in v.split(",") if x.strip()]
                except ValueError:
                    raise RsiError(f"bad --fidelity {v!r}: e.g. 1000,3000", "E_USAGE") from None
            kw[k] = v
        res = api.evolve(src, ov, space=a.space, name=a.name, out=a.out, resume=a.resume, force=a.force,
                         allow_code_change=a.allow_code_change, workers=a.workers, cache=cache, parts=parts,
                         on_event=log, **kw)
        return cmd, SCHEMAS[cmd], res
    if cmd == "bench":
        if a.action == "list":
            return "bench list", SCHEMAS["bench list"], api.bench_list()
        if a.action == "rank":
            if not a.files:
                raise RsiError("bench rank needs submission files", "E_USAGE")
            return "bench rank", SCHEMAS["bench rank"], api.bench_rank(a.files, top=a.top)
        if a.action is not None:
            raise RsiError(f"bench {a.action!r}: use list, rank FILE..., or flags only to run", "E_USAGE")
        api.setup_parts(parts)
        src, ov = config_sources(a, stdin)
        return cmd, SCHEMAS[cmd], api.bench(src, a.benchmark, a.workers, a.submit, overrides=ov, steps=a.steps, cache=cache,
                                           parts=parts, on_event=log)
    if cmd == "complexity":
        api.setup_parts(parts)
        src, ov = config_sources(a, stdin)
        ladder = None
        if a.ladder:
            try:
                ladder = [int(x) for x in a.ladder.split(",") if x.strip()]
            except ValueError:
                raise RsiError(f"bad --ladder {a.ladder!r}: comma-separated integers", "E_USAGE") from None
        return cmd, SCHEMAS[cmd], api.complexity(src, a.family, api.parse_seeds(a.sizes, "--sizes"), a.threshold,
                                                ladder, a.knob,
                                                api.parse_seeds(a.seeds), a.steps, a.workers, overrides=ov,
                                                cache=cache, parts=parts, on_event=log)
    if cmd == "status":
        return cmd, SCHEMAS[cmd], api.status(a.target, history=a.history)
    if cmd == "wait":
        return cmd, SCHEMAS[cmd], api.wait(a.target, timeout=a.timeout)
    if cmd == "stop":
        return cmd, SCHEMAS[cmd], api.stop(a.target, now=a.now, wait=a.wait, timeout=a.timeout)
    if cmd == "leaderboard":
        return cmd, SCHEMAS[cmd], api.leaderboard(a.targets, top=a.top, sort=a.sort, pareto=a.pareto)
    if cmd == "export":
        return cmd, SCHEMAS[cmd], api.export(a.target, rank=a.rank, genome_id=a.genome_id, out=a.out, steps=a.steps,
                                            open=a.open, parts=parts, dry_run=a.dry_run)
    if cmd == "open":
        api.setup_parts(parts)
        return cmd, SCHEMAS[cmd], api.open_ui(a.target, opengl=a.opengl, dry_run=a.dry_run)
    if cmd == "runs":
        rc = a.runs_cmd
        if rc is None:
            raise RsiError("runs needs list | query | show | stats", "E_USAGE")
        label = f"runs {rc}"
        if rc == "list":
            return label, SCHEMAS[label], api.runs_list(kind=a.kind, limit=a.limit)
        if rc == "query":
            return label, SCHEMAS[label], api.runs_query(where=a.where, sort=a.sort, limit=a.limit, fields=a.fields,
                                                         tag=a.tag, run=a.run, objective=a.objective)
        if rc == "show":
            return label, SCHEMAS[label], api.runs_show(a.id, curve=a.curve)
        return label, SCHEMAS[label], api.runs_stats(group_by=a.group_by, where=a.where, metric=a.metric)
    if cmd == "replay":
        return cmd, SCHEMAS[cmd], api.replay(a.trial_id, workers=a.workers, parts=parts)
    if cmd == "doctor":
        return cmd, SCHEMAS[cmd], api.doctor(golden=a.golden, record_golden=a.record_golden, calibrate=a.calibrate,
                                            parts=parts)
    if cmd == "gp":
        if a.gp_cmd != "check":
            raise RsiError("gp needs: check EXPR", "E_USAGE")
        return "gp check", SCHEMAS["gp check"], api.gp_check(a.expr)
    raise RsiError(f"unknown command {cmd!r}", "E_USAGE", allowed=list(COMMANDS))


# ---------------------------------------------------------------- main
def _meta():
    """Envelope meta. code_fp / parts are the same for every command: a command that loaded no
    parts (status, runs show, ...) reports the default parts a training command would load."""
    try:
        from nncore import registry
        from nncore.run import code_fingerprint, environment

        from . import api
        env = environment()
        parts = env["parts"]
        if not parts and not registry._lazy_done:
            parts = list(registry.DEFAULT_PLUGINS)
        return {"rsi": api.__version__, "nncore": env["nncore"], "torch": env["torch"], "numpy": env["numpy"],
                "python": env["python"], "platform": env["platform"], "code_fp": code_fingerprint(parts),
                "parts": parts}
    except Exception:
        return {}


def _serve_repl(a, g, stdin, out):
    from . import api
    from .repl import serve
    api.setup_parts(g["parts"] or None)
    src, ov = config_sources(a, None)
    cfg, _, _ = api.resolve_config(src, ov)
    return serve(cfg, stdin, out)


def _join_dash_values(argv):
    """'--sort -fitness' -> '--sort=-fitness' (argparse would read -fitness as a flag)."""
    out, i = [], 0
    while i < len(argv):
        if argv[i] == "--sort" and i + 1 < len(argv) and argv[i + 1].startswith("-") and not argv[i + 1].startswith("--"):
            out.append(f"--sort={argv[i + 1]}")
            i += 2
            continue
        out.append(argv[i])
        i += 1
    return out


def _full_trials(result):
    """--full: the sweep's trial rows carry their full config."""
    from .store import Store
    with Store() as st:
        for t in result.get("trials") or []:
            t["config"] = st.trial(t["id"])["config"]


def _sigterm(*_):
    raise KeyboardInterrupt  # flush partial results like Ctrl-C (exit 130)


def main(argv=None, *, out=None, stdin=None):
    """Run one command; returns the exit code. out=None: install the stdout guard and write
    to the real stdout (process use); else write the document to `out` (in-process use)."""
    from . import io
    in_process = out is not None
    if not in_process:
        out = io.install_guard()
    stdin = stdin if stdin is not None else sys.stdin
    argv = list(sys.argv[1:] if argv is None else argv)
    t0 = time.perf_counter()
    g = {"format": "json", "pretty": False, "unicode": False}
    command, schema, warnings = (argv[0] if argv and not argv[0].startswith("-") else None), "rsi/error@1", []
    old_store = os.environ.get("RSI_STORE")
    redirect = contextlib.redirect_stdout(sys.stderr) if in_process else contextlib.nullcontext()
    if not in_process and hasattr(signal, "SIGTERM"):
        try:
            signal.signal(signal.SIGTERM, _sigterm)
        except (ValueError, OSError):
            pass
    try:
        with redirect:
            from . import api
            with api.collect_warnings() as warnings:
                try:
                    g, rest = split_globals(argv)
                    if g["unicode"] and not in_process:
                        io.set_unicode(out)
                    if g["store"]:
                        os.environ["RSI_STORE"] = str(Path(g["store"]).resolve())
                    a = build_parser().parse_args(_join_dash_values(rest))
                    if a.command is None:
                        raise RsiError("no command given", "E_USAGE", allowed=list(COMMANDS),
                                       hint="python -m rsi describe --section commands")
                    command = a.command
                    if command == "repl":
                        return _serve_repl(a, g, stdin, out)
                    command, schema, result = dispatch(a, g, argv, stdin, out)
                    if g["full"] and schema == "rsi/sweep@1":
                        _full_trials(result)
                    doc = {"ok": True, "command": command, "schema": schema, "result": result}
                    code = 0
                except _Help as h:
                    sys.stderr.write(h.text)
                    doc = {"ok": True, "command": command or "help", "schema": "rsi/help@1", "result": {"usage": h.text}}
                    code = 0
                except BaseException as e:  # noqa: BLE001  (every failure becomes an envelope)
                    if isinstance(e, SystemExit):
                        raise
                    err = as_rsi_error(e)
                    doc = {"ok": False, "command": command, "schema": "rsi/error@1", "error": err.to_dict()}
                    if getattr(e, "result", None) is not None:
                        doc["result"] = e.result
                    code = err.exit
                    sys.stderr.write(f"rsi: error {err.code}: {err}\n")
            doc["warnings"] = list(warnings)
            doc["meta"] = _meta()
            doc["elapsed_s"] = round(time.perf_counter() - t0, 3)
            nonfinite = []
            text = io.to_json(doc, pretty=g.get("pretty", False), ascii=not g.get("unicode", False), nonfinite=nonfinite)
            if nonfinite:
                doc["warnings"].append({"code": "W_NONFINITE", "message": f"non-finite values became null at "
                                                                          f"{', '.join(nonfinite[:5])}",
                                        "paths": nonfinite[:50]})
                text = io.to_json(doc, pretty=g.get("pretty", False), ascii=not g.get("unicode", False))
            if g.get("format") == "text":
                text = io.render_text(json.loads(text))
            io.emit(text, out)
            return code
    finally:
        if in_process:
            if old_store is None:
                os.environ.pop("RSI_STORE", None)
            else:
                os.environ["RSI_STORE"] = old_store


def invoke(argv, stdin=None):
    """In-process run for tests and scripts: (exit code, parsed JSON document | text)."""
    import io as _io
    buf = _io.StringIO()
    code = main(list(argv), out=buf, stdin=_io.StringIO(stdin) if isinstance(stdin, str) else stdin)
    text = buf.getvalue()
    try:
        return code, json.loads(text)
    except ValueError:
        return code, text
