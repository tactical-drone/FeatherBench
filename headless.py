"""
headless.py // train without any window, for quick A/B checks of a part.

    python headless.py --dataset "Spiral (2 arms)" --model mlp --layers 8:sin,8 --steps 3000
    python headless.py --activation relu --compare activation=tanh,relu,gelu
    python headless.py --list              (every registered part per compartment)

Every Config field is a flag (--n_points or --n-points). --compare FIELD=a,b,c
runs one session per value with everything else equal and prints the results
side by side. Exit codes: 2 usage, 3 bad config, 1 anything else.

For scripts and AI agents (one JSON document on stdout, sweeps, search):
    python headless.py <command> ...       same as  python -m rsi <command> ...
    python headless.py describe            start here; see AGENTS.md
"""
import argparse
import os
import sys
import time
from dataclasses import fields


def _rsi_command(argv):
    """True when argv belongs to the rsi console: it starts with a bare word (a command; the
    legacy flags take no positionals, so a mistyped command still gets rsi's JSON E_USAGE
    envelope) or with an rsi global flag (--format, --store, ...)."""
    if not argv:
        return False
    if not argv[0].startswith("-"):
        return True
    from rsi.cli import GLOBAL_FLAG, GLOBAL_VALUE
    return argv[0].partition("=")[0] in GLOBAL_FLAG + GLOBAL_VALUE


def parse_value(f, text):
    """Flag text -> the field's value (configio.coerce_value, part names resolved)."""
    from nncore import configio
    if f.name == "extra":
        raise SystemExit("set 'extra' with: python headless.py run --set extra.NAME=VALUE")
    v = configio.coerce_value(f.name, text)
    if f.name == "features":
        return tuple(_resolve(f.name, t) for t in v)
    if f.name in configio.FIELD_REGISTRY:
        return _resolve(f.name, v)
    return v


def _resolve(field, name):
    from nncore import configio
    canonical, warn = configio.resolve_name(field, name)
    if warn:
        print(f"note: {warn['message']}", file=sys.stderr)
    return canonical


def run(cfg, steps, report_every):
    from nncore import Session
    s = Session(cfg)
    t0 = time.perf_counter()
    done = 0
    report_every = report_every if report_every > 0 else (steps or 1)
    while done < steps:
        n = min(report_every, steps - done)
        s.train(n)
        done += n
        ev = s.evaluate()
        print(f"  step {done:6d}  train {fmt(ev['train'])}   test {fmt(ev['test'])}")
    return s, s.evaluate(), time.perf_counter() - t0


def fmt(d):
    return "  ".join(f"{k} {v:.4f}" for k, v in d.items())


def _fail(code, msg):
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


def _config_error(e):
    msg = str(e)
    if getattr(e, "did_you_mean", None):
        msg += f" (did you mean: {', '.join(map(repr, e.did_you_mean))}?)"
    return msg


def _inert_warnings(cfg, set_fields, said):
    from nncore import Config, schema
    try:
        inert = schema.inert_fields(cfg)
    except Exception:
        return
    default = Config()
    for f in sorted(set(set_fields) & inert):
        if getattr(cfg, f) != getattr(default, f) and (f, cfg.model) not in said:
            said.add((f, cfg.model))
            print(f"warning: {f} is not read by model '{cfg.model}' (ignored)", file=sys.stderr)


def legacy_main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)  # part names use →, ², ·; show progress live
    except (AttributeError, ValueError):
        pass
    try:
        import my_parts  # noqa: F401  (registers your parts)
    except Exception as e:
        import traceback
        print("error in my_parts.py (your workbench):", file=sys.stderr)
        print("".join(traceback.format_exception(e)[-3:]), file=sys.stderr, end="")
        sys.exit(1)
    from nncore import REGISTRIES, Config, configio

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    by_name = {f.name: f for f in fields(Config)}
    for f in fields(Config):
        if f.name != "extra":
            names = [f"--{f.name}"] + ([f"--{f.name.replace('_', '-')}"] if "_" in f.name else [])
            ap.add_argument(*names, dest=f.name, help=f"default: {f.default!r}")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--report-every", "--report_every", dest="report_every", type=int, default=500)
    ap.add_argument("--compare", help="FIELD=v1,v2,...  (use ';' to separate values containing commas)")
    ap.add_argument("--list", action="store_true", help="list registered parts and exit")
    args = ap.parse_args(argv)
    if args.steps < 0:
        ap.error("--steps must be >= 0")

    if args.list:
        for field, reg in REGISTRIES.items():
            print(f"{field:12s} {reg.signature}")
            for name in reg:
                print(f"               - {name}")
        return 0

    try:
        base = {k: parse_value(by_name[k], v) for k, v in vars(args).items()
                if k in by_name and v is not None}
    except configio.ConfigError as e:
        _fail(3, _config_error(e))

    runs = [("", base)]
    set_fields = list(base)
    if args.compare:
        key, eq, vals = args.compare.partition("=")
        key = key.strip().replace("-", "_")
        if not eq:
            ap.error(f"--compare {args.compare!r}: expected FIELD=v1,v2,...")
        if key not in by_name or key == "extra":
            ap.error(f"--compare: unknown field '{key}'")
        sep = ";" if ";" in vals else ","
        values = [v.strip() for v in vals.split(sep) if v.strip()]
        if not values:
            ap.error("--compare: no values")
        try:
            runs = [(f"{key}={v}", {**base, key: parse_value(by_name[key], v)}) for v in values]
        except configio.ConfigError as e:
            _fail(3, _config_error(e))
        set_fields.append(key)

    configs, said = [], set()
    for label, overrides in runs:  # validate every run before the first one starts
        cfg = Config(**overrides)
        try:
            configio.check_config(cfg)
        except configio.ConfigError as e:
            _fail(3, f"{label + ': ' if label else ''}{_config_error(e)}")
        _inert_warnings(cfg, set_fields, said)
        configs.append((label, cfg))

    results, failed = [], 0
    for label, cfg in configs:
        print(f"\n== {label or 'run'}")
        try:
            s, ev, secs = run(cfg, args.steps, args.report_every)
        except Exception as e:  # one broken run does not lose the others
            print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
            failed += 1
            continue
        print(f"  net {s.describe()}  params {s.n_params}  {secs:.2f}s")
        results.append((label or "run", ev))

    if len(results) > 1:
        print("\n== summary")
        for label, ev in results:
            print(f"  {label:32s} train {fmt(ev['train'])}   test {fmt(ev['test'])}")
    return 1 if failed else 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if _rsi_command(argv):  # python headless.py <command> == python -m rsi <command>
        from rsi.cli import main as rsi_main
        return rsi_main(argv)
    try:
        return legacy_main(argv)
    except BrokenPipeError:  # e.g. python headless.py --list | head
        try:
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        except (OSError, ValueError):
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
