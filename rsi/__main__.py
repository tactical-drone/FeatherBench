"""python -m rsi <command>: the rsi console (see rsi/cli.py). Part of nn-playground (AGPL-3.0; see COMMERCIAL.md)."""
import sys

from rsi.cli import main

if __name__ == "__main__":  # spawned workers re-import this module as __mp_main__
    sys.exit(main())
