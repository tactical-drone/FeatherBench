"""
NN Playground // tiny neural nets, trained live.

Launcher only. The neural-net code lives in nncore/, the window in ui/,
and your own parts in my_parts.py.

Run:  python nn_playground.py            (add --opengl to try the GL viewport)
"""
import argparse
import sys

import my_parts  # noqa: F401  (registers your parts before the UI lists them)
from ui import run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opengl", action="store_true", help="use the OpenGL viewport")
    args = ap.parse_args()
    sys.exit(run(opengl=args.opengl))


if __name__ == "__main__":
    main()
