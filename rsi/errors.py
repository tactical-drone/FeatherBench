"""
errors: RsiError (one error code + exit code per failure) and the code tables.

    raise RsiError("no run named 's5a'", "E_NOT_FOUND", hint="python -m rsi runs list")

Part of nn-playground. AGPL-3.0; for other licensing see COMMERCIAL.md.
"""
import traceback

EXIT_CODES = {
    "E_INTERNAL": 1, "E_USAGE": 2,
    **{c: 3 for c in ("E_UNKNOWN_FIELD", "E_UNKNOWN_PART", "E_AMBIGUOUS", "E_BAD_VALUE", "E_OUT_OF_RANGE",
                      "E_BAD_LAYERS", "E_BAD_SPACE", "E_BAD_SETTINGS", "E_BAD_QUERY", "E_BAD_EXPR",
                      "E_UNSUPPORTED")},
    "E_RUN_FAILED": 4,
    **{c: 5 for c in ("E_NOT_FOUND", "E_AMBIGUOUS_ID", "E_EXISTS", "E_LOCKED", "E_RESUME_MISMATCH",
                      "E_CODE_CHANGED", "E_PARTS_MISSING", "E_IO")},
    "E_TIMEOUT": 7, "E_NOT_REPRODUCIBLE": 10, "E_INTERRUPTED": 130,
}
EXIT_MEANING = {"0": "ok (also partial sweeps, diverged/timeout cells, stopped searches)", "1": "internal error",
                "2": "usage", "3": "invalid input", "4": "run failed (every cell error/diverged)",
                "5": "run dir / store / file problem", "7": "wait timed out", "10": "replay not reproducible",
                "130": "interrupted"}
ERROR_HELP = {
    "E_INTERNAL": "a bug or an unexpected exception; see error.traceback",
    "E_USAGE": "bad flags or subcommand; see python -m rsi describe --section commands",
    "E_UNKNOWN_FIELD": "not a Config field; see did_you_mean / describe --section levers",
    "E_UNKNOWN_PART": "no part by that name; see did_you_mean / describe --lever FIELD (or load it with --parts)",
    "E_AMBIGUOUS": "the name matches several parts; use one of did_you_mean",
    "E_BAD_VALUE": "wrong type (a number, a name, an object...)",
    "E_OUT_OF_RANGE": "outside the field's bounds (describe --lever FIELD)",
    "E_BAD_LAYERS": "layers grammar: '' | W[:ACT](,W[:ACT])*, W in 1..512",
    "E_BAD_SPACE": "invalid space file (schema space)",
    "E_BAD_SETTINGS": "not a settings doc / config / trial record, or a newer version",
    "E_BAD_QUERY": "--where is PATH OP VALUE, OP in == != < <= > >= ~ in",
    "E_BAD_EXPR": "bad GP expression (a gp:/gpl: activation or gp check); python -m rsi gp check EXPR says why",
    "E_UNSUPPORTED": "the feature is not available in this build",
    "E_RUN_FAILED": "every cell had status error or diverged; read cells[].error",
    "E_NOT_FOUND": "no such run dir, trial id or file",
    "E_AMBIGUOUS_ID": "the id prefix matches several trials; give more characters",
    "E_EXISTS": "the run dir exists; pick another --name or pass --force",
    "E_LOCKED": "another live process holds the run dir",
    "E_RESUME_MISMATCH": "only budget / worker flags may change on --resume",
    "E_CODE_CHANGED": "nncore or parts changed since the run started; --allow-code-change",
    "E_PARTS_MISSING": "the file needs part modules that are not loaded; pass --parts",
    "E_IO": "file system error",
    "E_TIMEOUT": "wait --timeout expired; result still carries the status",
    "E_NOT_REPRODUCIBLE": "replay differs from the stored cells; see details.diffs",
    "E_INTERRUPTED": "Ctrl-C / SIGTERM; partial results were flushed",
}


class RsiError(Exception):
    """A console failure with a stable code; .exit is the process exit code."""
    def __init__(self, message, code="E_INTERNAL", *, field=None, value=None, did_you_mean=None, allowed=None,
                 hint=None, details=None, exit=None):
        super().__init__(message)
        self.code = code
        self.exit = exit if exit is not None else EXIT_CODES.get(code, 1)
        self.field, self.value, self.hint = field, value, hint
        self.did_you_mean, self.allowed = list(did_you_mean or []), allowed
        self.details = dict(details or {})
        self.tb = None

    def to_dict(self):
        d = {"code": self.code, "exit": self.exit, "message": str(self), "field": self.field, "value": self.value,
             "did_you_mean": self.did_you_mean, "allowed": self.allowed, "hint": self.hint, "traceback": self.tb}
        if self.details:
            d["details"] = self.details
        return d


def as_rsi_error(e):
    """Any exception -> RsiError (ConfigError keeps its code; the rest is E_INTERNAL with a traceback)."""
    if isinstance(e, RsiError):
        return e
    if isinstance(e, KeyboardInterrupt):
        return RsiError("interrupted", "E_INTERRUPTED")
    code = getattr(e, "code", None)
    if isinstance(code, str) and code in EXIT_CODES:  # nncore.configio.ConfigError
        return RsiError(str(e), code, field=getattr(e, "field", None), value=getattr(e, "value", None),
                        did_you_mean=getattr(e, "did_you_mean", None), allowed=getattr(e, "allowed", None),
                        hint=getattr(e, "hint", None))
    if isinstance(e, (FileNotFoundError, IsADirectoryError)):
        return RsiError(str(e), "E_NOT_FOUND")
    if isinstance(e, OSError):
        return RsiError(str(e), "E_IO")
    err = RsiError(f"{type(e).__name__}: {e}", "E_INTERNAL")
    err.tb = "".join(traceback.format_exception(e)[-20:])
    return err
