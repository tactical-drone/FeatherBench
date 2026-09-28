"""Qt / pyqtgraph front end for nncore. Nothing in nncore imports from here."""
from .app import Playground, run

__all__ = ["Playground", "run"]
