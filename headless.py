"""
headless.py // train without any window, for quick A/B checks of a part.

    python headless.py --dataset "Spiral (2 arms)" --layers 8:sin,8 --steps 3000
    python headless.py --activation relu --compare activation=tanh,relu,gelu
    python headless.py --list              (every registered part per compartment)

Every Config field is a flag. --compare FIELD=a,b,c runs one session per value
with everything else equal and prints the results side by side.
"""
import argparse
import sys
import time
from dataclasses import fields

import my_parts  # noqa: F401  (registers your parts)
from nncore import REGISTRIES, Config, Session


def parse_value(f, text):
    if f.name == "features":
        return tuple(t for t in text.split(",") if t)
    if f.name == "batch_size":
        return None if text in ("full", "none", "None") else int(text)
    if f.name == "extra":
        raise SystemExit("set 'extra' from code, not the command line")
    default = f.default
    return type(default)(text) if isinstance(default, (int, float)) else text


def run(cfg, steps, report_every):
    s = Session(cfg)
    t0 = time.perf_counter()
    done = 0
    while done < steps:
        n = min(report_every, steps - done)
        s.train(n)
        done += n
        ev = s.evaluate()
        print(f"  step {done:6d}  train {fmt(ev['train'])}   test {fmt(ev['test'])}")
    return s, s.evaluate(), time.perf_counter() - t0


def fmt(d):
    return "  ".join(f"{k} {v:.4f}" for k, v in d.items())


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # part names use →, ², · (Windows console is cp1252)
    ap =argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    by_name = {f.name: f for f in fields(Config)}
    for f in fields(Config):
        if f.name != "extra":
            ap.add_argument(f"--{f.name}", help=f"default: {f.default!r}")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--report-every", type=int, default=500)
    ap.add_argument("--compare", help="FIELD=v1,v2,...  (use ';' to separate values containing commas)")
    ap.add_argument("--list", action="store_true", help="list registered parts and exit")
    args = ap.parse_args()

    if args.list:
        for field, reg in REGISTRIES.items():
            print(f"{field:12s} {reg.signature}")
            for name in reg:
                print(f"               - {name}")
        return

    base = {k: parse_value(by_name[k], v) for k, v in vars(args).items()
            if k in by_name and v is not None}

    runs = [("", base)]
    if args.compare:
        key, _, vals = args.compare.partition("=")
        sep = ";" if ";" in vals else ","
        runs = [(f"{key}={v}", {**base, key: parse_value(by_name[key], v)})
                for v in vals.split(sep)]

    results = []
    for label, overrides in runs:
        cfg = Config(**overrides)
        print(f"\n== {label or 'run'}")
        s, ev, secs = run(cfg, args.steps, args.report_every)
        print(f"  net {s.describe()}  params {s.n_params}  {secs:.2f}s")
        results.append((label or "run", ev))

    if len(results) > 1:
        print("\n== summary")
        for label, ev in results:
            print(f"  {label:32s} train {fmt(ev['train'])}   test {fmt(ev['test'])}")


if __name__ == "__main__":
    main()
