"""
Registry: a named slot where interchangeable parts live.

Every compartment (datasets, activations, optimizers, ...) is one Registry.
Adding a part is one decorator; the UI and the headless runner list whatever
is registered, so new parts show up everywhere automatically.

    from nncore import ACTIVATIONS

    @ACTIVATIONS.register("swish2", doc="x * sigmoid(2x)")
    def swish2():
        return MySwish2()

Parts that live outside nncore (my_parts.py, agent_parts/...) register when
their module is imported. load_plugins() imports them; a lookup that misses
also tries DEFAULT_PLUGINS once, so Session(Config()) works without an
explicit `import my_parts` (spawned worker processes rely on this).
"""
import difflib
import importlib
import importlib.util
import inspect
import os
import sys
import warnings
from pathlib import Path

_MISSING = object()
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PLUGINS = tuple(p for p in os.environ.get("NNCORE_PLUGINS", "my_parts").split(os.pathsep) if p)
HINT = "(custom parts live in my_parts.py; call nncore.load_plugins())"

_loaded = []          # canonical module names, in load order
_lazy_done = False    # DEFAULT_PLUGINS tried once on the first miss


class ResolveError(ValueError):
    """Raised by a resolver for a name it owns but cannot build ('gp:foo(x)'): the lookup
    misses, and the reason (with an error code and a hint) replaces 'unknown <kind>'."""
    def __init__(self, message, code="E_BAD_VALUE", hint=None):
        super().__init__(message)
        self.code, self.hint = code, hint


class Registry(dict):
    def __init__(self, kind, signature=""):
        super().__init__()
        self.kind = kind
        self.signature = signature  # human-readable contract, shown in errors
        self.docs = {}              # name -> one-line description
        self.resolved = {}          # name -> obj made by a resolver (not listed in the dropdowns)
        self.rejected = {}          # name -> ResolveError from a resolver (why it is not a part)
        self._resolvers = []

    def register(self, name, obj=None, *, doc=None, override=True):
        """Use as @REG.register("name") or REG.register("name", obj). With
        override=False an existing entry from another module is kept (UserWarning)."""
        def put(o):
            old = dict.get(self, name)
            if old is not None and not override:
                if getattr(old, "__module__", None) != getattr(o, "__module__", None):
                    warnings.warn(f"{self.kind} '{name}' already registered by {getattr(old, '__module__', '?')}; "
                                  f"keeping it (override=False)", UserWarning, stacklevel=3)
                    return o
            self[name] = o
            text = doc
            if text is None and (inspect.isfunction(o) or inspect.isclass(o)):
                text = (inspect.getdoc(o) or "").strip().split("\n")[0]
            if text:
                self.docs[name] = text
            return o

        if obj is not None:
            return put(obj)
        return put

    def add_resolver(self, fn):
        """fn(name) -> obj | None, consulted after a plain lookup misses; a hit is
        cached in self.resolved (iteration, names() and the UI lists stay unchanged).
        fn may raise ResolveError to say why a name it owns is invalid (see rejection())."""
        self._resolvers.append(fn)

    def _resolve(self, name):
        if name in self.resolved:
            return self.resolved[name]
        for fn in self._resolvers:
            try:
                obj = fn(name)
            except ResolveError as e:
                if len(self.rejected) >= 256:
                    self.rejected.clear()
                self.rejected[name] = e
                continue
            if obj is not None:
                self.resolved[name] = obj
                return obj
        return _MISSING

    def rejection(self, name):
        """The ResolveError a resolver raised for name (after a miss), else None."""
        return self.rejected.get(name) if isinstance(name, str) else None

    def has(self, name):
        """True if get(name) would succeed without loading plugins."""
        return name in self or self._resolve(name) is not _MISSING

    def get(self, name, default=_MISSING):
        """REG.get(name) -> part, or ValueError with suggestions; REG.get(name, default)
        has dict semantics (resolvers are still consulted, plugins are not loaded)."""
        if name in self:
            return self[name]
        obj = self._resolve(name)
        if obj is not _MISSING:
            return obj
        if default is not _MISSING:
            return default
        if ensure_plugins():
            return self.get(name)
        rej = self.rejection(name)
        if rej is not None:
            raise ResolveError(str(rej), rej.code, rej.hint) from None
        raise ValueError(self.unknown_message(name)) from None

    def suggest(self, name, n=3):
        """Close matches for a misspelt name (case-insensitive first)."""
        names = list(self)
        low = {k.lower(): k for k in names}
        out = [low[name.lower()]] if isinstance(name, str) and name.lower() in low else []
        if isinstance(name, str):
            out += [low[m] for m in difflib.get_close_matches(name.lower(), list(low), n=n, cutoff=0.6)]
        return list(dict.fromkeys(out))[:n]

    def unknown_message(self, name):
        hint = self.suggest(name)
        mean = f" (did you mean: {', '.join(repr(h) for h in hint)}?)" if hint else ""
        return f"unknown {self.kind} '{name}'{mean} (have: {', '.join(self)}) {HINT}"

    def names(self):
        return list(self)

    def __repr__(self):
        return f"Registry({self.kind}: {', '.join(self)})"


def call_accepting(fn, *args, **knobs):
    """fn(*args, **the knobs fn accepts): all of them if fn takes **kwargs, none it doesn't name."""
    return fn(*args, **accepted(fn, knobs))


def accepted(fn, knobs):
    """The subset of `knobs` (dict) that fn's signature accepts."""
    if not knobs:
        return {}
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    if any(p.kind is p.VAR_KEYWORD for p in params.values()):
        return dict(knobs)
    return {k: v for k, v in knobs.items()
            if k in params and params[k].kind in (params[k].POSITIONAL_OR_KEYWORD, params[k].KEYWORD_ONLY)}


# ---------------------------------------------------------------- plugins
def _module_name(spec):
    """'my_parts' | 'agent_parts.x' | 'agent_parts/x.py' -> (module name, path if outside the repo)."""
    if not spec.endswith(".py"):
        return spec, None
    p = Path(spec)
    p = (p if p.is_absolute() else Path.cwd() / p).resolve()
    try:
        return ".".join(p.relative_to(REPO_ROOT).with_suffix("").parts), None
    except ValueError:
        return p.stem, p


def load_plugins(*modules):
    """Import part modules (names, or .py paths) so their parts register. Idempotent.
    No arguments = DEFAULT_PLUGINS. Returns the canonical names loaded so far for these."""
    out = []
    for spec in modules or DEFAULT_PLUGINS:
        name, path = _module_name(spec)
        if name not in sys.modules:
            if path is not None:
                s = importlib.util.spec_from_file_location(name, path)
                mod = importlib.util.module_from_spec(s)
                sys.modules[name] = mod
                try:
                    s.loader.exec_module(mod)
                except BaseException:
                    del sys.modules[name]
                    raise
            else:
                if str(REPO_ROOT) not in sys.path and importlib.util.find_spec(name.split(".")[0]) is None:
                    sys.path.append(str(REPO_ROOT))
                importlib.import_module(name)
        if name not in _loaded:
            _loaded.append(name)
        out.append(name)
    return out


def loaded_plugins():
    """Plugin modules loaded through load_plugins (or the lazy fallback), in order;
    my_parts also counts when someone did a plain `import my_parts`."""
    extra = [m for m in DEFAULT_PLUGINS if m in sys.modules and m not in _loaded]
    return list(_loaded) + extra


def disable_lazy_plugins():
    """Never auto-load DEFAULT_PLUGINS on a miss (e.g. the console's --parts none)."""
    global _lazy_done
    _lazy_done = True


def ensure_plugins():
    """Try DEFAULT_PLUGINS once per process; True if that newly loaded something."""
    global _lazy_done
    if _lazy_done:
        return False
    _lazy_done = True
    before = set(sys.modules)
    for m in DEFAULT_PLUGINS:
        try:
            load_plugins(m)
        except ModuleNotFoundError as e:
            if e.name != _module_name(m)[0].split(".")[0]:
                raise  # the plugin exists but imports something missing: say so
    return bool(set(sys.modules) - before)
