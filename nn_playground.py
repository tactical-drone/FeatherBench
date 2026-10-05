"""
NN Playground // tiny neural nets, trained live.

Launcher only. The neural-net code lives in nncore/, the window in ui/,
and your own parts in my_parts.py.

Run:  python nn_playground.py            (add --opengl to try the GL viewport)
      python nn_playground.py --load my_settings.json
      python nn_playground.py --fresh    (ignore the settings remembered from last time)
"""
import argparse
import sys

if sys.version_info < (3, 10):
    sys.exit("NN Playground needs Python 3.10 or newer (you have %d.%d)." % sys.version_info[:2])

try:
    import my_parts  # noqa: F401  (registers your parts before the UI lists them)
    from ui import run
except ModuleNotFoundError as e:
    if e.name in ("torch", "numpy", "pyqtgraph", "PyQt6"):
        sys.exit(f"Missing package '{e.name}'. Install everything with:\n"
                 f"    pip install -r requirements.txt")
    raise


def main():
    from ui.app import VERSION
    ap = argparse.ArgumentParser(description="Tiny neural nets, trained live.")
    ap.add_argument("--opengl", action="store_true", help="use the OpenGL viewport")
    ap.add_argument("--load", metavar="FILE", help="start from a settings .json (File > Save settings)")
    ap.add_argument("--fresh", action="store_true", help="start from the defaults, not last session's settings")
    ap.add_argument("--version", action="version", version=f"NN Playground {VERSION}")
    ap.add_argument("--smoke-test", type=int, metavar="STEPS", help=argparse.SUPPRESS)  # release CI: train, exit 0
    args = ap.parse_args()
    sys.exit(run(opengl=args.opengl, fresh=args.fresh, settings_file=args.load, smoke_test=args.smoke_test))


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # the packaged .exe: garage jobs start worker processes
    main()
