"""
gp: expression activations for `evolve --gp-activations` (P2, opt-in).

    gp:mul(x;sin(mul(3;x)))     == x * sin(3x)     ('gpl:' = the constants are learnable)
    python -m rsi gp check "gp:mul(x;sin(mul(3;x)))"

Syntax: prefix form op(a;b), no spaces, commas or further colons, so a name
survives parse_layers ("8:gp:sin(x),4"). Primitives: x; constants (numbers,
pi); unary neg abs sq sin cos tanh sigmoid softplus relu gauss; binary add sub
mul max min pdiv (protected a*b/(b^2+1e-6)). Trees are interpreted with torch
ops (no eval/exec). Importing this module (or install_resolver()) lets
ACTIVATIONS.get("gp:...") build them, so Config / settings files may name them.

Part of nn-playground. AGPL-3.0; for other licensing see COMMERCIAL.md.
"""
import functools
import hashlib
import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from nncore.activations import ACTIVATIONS
from nncore.registry import Registry, ResolveError

from .errors import RsiError

PREFIXES = ("gp:", "gpl:")
CONSTS = (0.5, 1.0, 2.0, 3.0, math.pi)
PROBE = torch.linspace(-5.0, 5.0, 64)
GP_PRIMITIVES = Registry("gp primitive", "name -> (arity, torch fn, python fn, infix format)")


def _pdiv_t(a, b):
    return a * b / (b * b + 1e-6)


def _pdiv_p(a, b):
    return a * b / (b * b + 1e-6)


def _sp(v):
    return math.log1p(math.exp(-abs(v))) + max(v, 0.0)


def _sig(v):
    return 1 / (1 + math.exp(-v)) if v >= 0 else math.exp(v) / (1 + math.exp(v))


for _n, _t, _p, _fmt in (
        ("neg", torch.neg, lambda a: -a, "-{0}"), ("abs", torch.abs, abs, "|{0}|"),
        ("sq", torch.square, lambda a: a * a, "({0})^2"), ("sin", torch.sin, math.sin, "sin({0})"),
        ("cos", torch.cos, math.cos, "cos({0})"), ("tanh", torch.tanh, math.tanh, "tanh({0})"),
        ("sigmoid", torch.sigmoid, _sig, "sigmoid({0})"), ("softplus", F.softplus, _sp, "softplus({0})"),
        ("relu", torch.relu, lambda a: max(a, 0.0), "relu({0})"),
        ("gauss", lambda a: torch.exp(-a * a), lambda a: math.exp(-a * a), "gauss({0})")):
    GP_PRIMITIVES.register(_n, (1, _t, _p, _fmt))
for _n, _t, _p, _fmt in (
        ("add", torch.add, lambda a, b: a + b, "({0} + {1})"), ("sub", torch.sub, lambda a, b: a - b, "({0} - {1})"),
        ("mul", torch.mul, lambda a, b: a * b, "{0}*{1}"), ("max", torch.maximum, max, "max({0}, {1})"),
        ("min", torch.minimum, min, "min({0}, {1})"), ("pdiv", _pdiv_t, _pdiv_p, "{0}/{1}")):
    GP_PRIMITIVES.register(_n, (2, _t, _p, _fmt))
UNARY = tuple(n for n, v in GP_PRIMITIVES.items() if v[0] == 1)
BINARY = tuple(n for n, v in GP_PRIMITIVES.items() if v[0] == 2)


class GPError(ValueError):
    pass


class Node:
    """op 'x', 'c' (value) or a primitive name with args."""
    __slots__ = ("op", "args", "value")

    def __init__(self, op, args=(), value=None):
        self.op, self.args, self.value = op, tuple(args), value

    def __eq__(self, other):
        return isinstance(other, Node) and to_prefix(self) == to_prefix(other)

    def __hash__(self):
        return hash(to_prefix(self))

    def __repr__(self):
        return f"Node({to_prefix(self)})"


# ---------------------------------------------------------------- text
_TOKEN = re.compile(r"\s*(?:(?P<num>-?\d+(?:\.\d+)?(?:e-?\d+)?)|(?P<name>[a-z]+)|(?P<sym>[();]))")
MAX_TEXT, MAX_NEST = 2000, 40  # parse limits (far above --gp-max-nodes / --gp-max-depth)


def is_gp(name):
    return isinstance(name, str) and name.startswith(PREFIXES)


def parse(name):
    """'gp:mul(x;sin(x))' (prefix optional) -> Node; GPError on bad syntax."""
    text = name.split(":", 1)[1] if is_gp(name) else str(name)
    if not text or any(c in text for c in ",: "):
        raise GPError(f"bad GP expression {name!r}: no spaces, commas or colons inside")
    if len(text) > MAX_TEXT:
        raise GPError(f"GP expression too long ({len(text)} characters, at most {MAX_TEXT})")
    nest = 0
    for c in text:
        nest += (c == "(") - (c == ")")
        if nest > MAX_NEST:
            raise GPError(f"GP expression nested too deeply (more than {MAX_NEST} levels)")
    toks, pos = [], 0
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m or m.end() == pos:
            raise GPError(f"bad GP expression {name!r}: unexpected {text[pos:pos + 8]!r}")
        toks.append(m.group("num") or m.group("name") or m.group("sym"))
        pos = m.end()
    node, i = _expr(toks, 0, name)
    if i != len(toks):
        raise GPError(f"bad GP expression {name!r}: trailing {''.join(toks[i:])!r}")
    return node


def _expr(toks, i, name):
    if i >= len(toks):
        raise GPError(f"bad GP expression {name!r}: unexpected end")
    t = toks[i]
    if t == "x":
        return Node("x"), i + 1
    if t == "pi":
        return Node("c", value=math.pi), i + 1
    if re.fullmatch(r"-?\d.*", t):
        return Node("c", value=float(t)), i + 1
    if t not in GP_PRIMITIVES:
        raise GPError(f"bad GP expression {name!r}: unknown op {t!r} (have x, numbers, pi, "
                      f"{', '.join(GP_PRIMITIVES)})")
    arity = GP_PRIMITIVES[t][0]
    if i + 1 >= len(toks) or toks[i + 1] != "(":
        raise GPError(f"bad GP expression {name!r}: {t} needs '('")
    args, j = [], i + 2
    for k in range(arity):
        a, j = _expr(toks, j, name)
        args.append(a)
        want = ";" if k < arity - 1 else ")"
        if j >= len(toks) or toks[j] != want:
            raise GPError(f"bad GP expression {name!r}: {t} takes {arity} argument(s) separated by ';'")
        j += 1
    return Node(t, args), j


def _num(v):
    if v == math.pi:
        return "pi"
    if float(v).is_integer() and abs(v) < 1e6:
        return str(int(v))
    return f"{v:.6g}"


def to_prefix(n):
    if n.op == "x":
        return "x"
    if n.op == "c":
        return _num(n.value)
    return f"{n.op}({';'.join(to_prefix(a) for a in n.args)})"


def to_name(n, learnable=False):
    return ("gpl:" if learnable else "gp:") + to_prefix(n)


def to_infix(n):
    if n.op == "x":
        return "x"
    if n.op == "c":
        return _num(n.value)
    return GP_PRIMITIVES[n.op][3].format(*(to_infix(a) for a in n.args))


def size(n):
    return 1 + sum(size(a) for a in n.args)


def depth(n):
    return 1 + max((depth(a) for a in n.args), default=0)


def _paths(n, path=()):
    yield path, n
    for i, a in enumerate(n.args):
        yield from _paths(a, path + (i,))


def _replace(n, path, sub):
    if not path:
        return sub
    args = list(n.args)
    args[path[0]] = _replace(args[path[0]], path[1:], sub)
    return Node(n.op, args, n.value)


# ---------------------------------------------------------------- simplify
def _const(n):
    return n.op == "c"


def simplify(n):
    """Constant folding plus a few identities (x+0, x*1, x*0, --x, max(a;a), ...)."""
    if n.op in ("x", "c"):
        return n
    args = [simplify(a) for a in n.args]
    op = n.op
    if all(_const(a) for a in args):
        try:
            v = GP_PRIMITIVES[op][2](*(a.value for a in args))
            if math.isfinite(v):
                return Node("c", value=float(v))
        except (OverflowError, ValueError, ZeroDivisionError):
            pass
    a = args[0]
    b = args[1] if len(args) > 1 else None
    zero, one = (lambda m: _const(m) and m.value == 0), (lambda m: _const(m) and m.value == 1)
    if op == "add" and zero(b):
        return a
    if op == "add" and zero(a):
        return b
    if op == "sub" and zero(b):
        return a
    if op == "sub" and a == b:
        return Node("c", value=0.0)
    if op == "mul" and (zero(a) or zero(b)):
        return Node("c", value=0.0)
    if op == "mul" and one(b):
        return a
    if op == "mul" and one(a):
        return b
    if op == "pdiv" and one(b):
        return a
    if op in ("max", "min") and a == b:
        return a
    if op == "neg" and a.op == "neg":
        return a.args[0]
    if op == "abs" and a.op in ("abs", "sq", "gauss", "sigmoid", "softplus", "relu"):
        return a
    return Node(op, args)


# ---------------------------------------------------------------- evaluation
def evaluate(n, x, params=None):
    """The tree applied to tensor x (params: {path: nn.Parameter} for learnable constants)."""
    def go(m, path):
        if m.op == "x":
            return x
        if m.op == "c":
            if params is not None and path in params:
                return params[path] * torch.ones_like(x)
            return torch.full_like(x, m.value)
        return GP_PRIMITIVES[m.op][1](*(go(a, path + (i,)) for i, a in enumerate(m.args)))
    return go(n, ())


class ExprActivation(nn.Module):
    """An activation from a GP tree; learnable=True turns every constant into an
    nn.Parameter initialised to its value (no RNG use)."""

    def __init__(self, tree, learnable=False):
        super().__init__()
        self.tree = parse(tree) if isinstance(tree, str) else tree
        self.consts = nn.ParameterList()
        self._paths = []
        if learnable:
            for path, m in _paths(self.tree):
                if m.op == "c":
                    self._paths.append(path)
                    self.consts.append(nn.Parameter(torch.tensor(float(m.value))))

    def forward(self, x):
        params = dict(zip(self._paths, self.consts)) if self._paths else None
        return evaluate(self.tree, x, params)

    def extra_repr(self):
        return to_infix(self.tree)


def sanity(n):
    """(ok, reason): rejects non-finite, constant, huge (|f| > 1e4) and zero-derivative functions."""
    x = PROBE.clone().requires_grad_(True)
    try:
        y = evaluate(n, x)
    except Exception as e:  # noqa: BLE001
        return False, f"evaluation failed: {e}"
    if not torch.isfinite(y).all():
        return False, "non-finite values on [-5, 5]"
    if y.abs().max().item() > 1e4:
        return False, "|f| > 1e4 on [-5, 5]"
    if (y.max() - y.min()).item() < 1e-6:
        return False, "constant function"
    (g,) = torch.autograd.grad(y.sum(), x, allow_unused=True)
    if g is None or g.abs().max().item() < 1e-8:
        return False, "derivative is zero everywhere"
    return True, "ok"


def semantic_hash(n):
    """12 hex over the values on 64 probe points (rounded): equal functions collide."""
    with torch.no_grad():
        y = evaluate(n, PROBE)
    vals = ",".join(f"{v:.5g}" for v in y.tolist())
    return hashlib.sha1(vals.encode()).hexdigest()[:12]


# ---------------------------------------------------------------- variation
def _terminal(rng):
    if rng.random() < 0.75:
        return Node("x")
    if rng.random() < 0.6:
        return Node("c", value=rng.choice(CONSTS))
    return Node("c", value=round(rng.uniform(-2, 2) * 4) / 4)


def random_tree(rng, max_depth, method="grow"):
    """A random tree of depth <= max_depth ('full': every branch reaches max_depth)."""
    if max_depth <= 1 or (method == "grow" and rng.random() < 0.3):
        return _terminal(rng)
    op = rng.choice(UNARY + BINARY)
    return Node(op, [random_tree(rng, max_depth - 1, method) for _ in range(GP_PRIMITIVES[op][0])])


def _fits(n, max_depth, max_nodes):
    return depth(n) <= max_depth and size(n) <= max_nodes and any(m.op == "x" for _, m in _paths(n))


def mutate(n, rng, max_depth=4, max_nodes=15):
    """Subtree replacement (p 0.5) or a point mutation (op of the same arity, or a constant)."""
    for _ in range(10):
        paths = list(_paths(n))
        path, node = rng.choice(paths)
        if rng.random() < 0.5:
            new = _replace(n, path, random_tree(rng, max(1, max_depth - len(path))))
        elif node.op in ("x", "c"):
            new = _replace(n, path, _terminal(rng))
        else:
            same = [o for o in GP_PRIMITIVES if GP_PRIMITIVES[o][0] == len(node.args) and o != node.op]
            new = _replace(n, path, Node(rng.choice(same), node.args))
        if new != n and _fits(new, max_depth, max_nodes):
            return new
    return n


def crossover(a, b, rng, max_depth=4, max_nodes=15):
    """a with one random subtree replaced by a random subtree of b."""
    for _ in range(10):
        pa, _ = rng.choice(list(_paths(a)))
        _, sb = rng.choice(list(_paths(b)))
        new = _replace(a, pa, sb)
        if _fits(new, max_depth, max_nodes):
            return new
    return a


# ---------------------------------------------------------------- registry hook
_INSTALLED = []


def _resolve(name):
    """ACTIVATIONS resolver; a malformed expression raises ResolveError (E_BAD_EXPR), so
    run / check report the parse error instead of a bare 'unknown activation'."""
    if not is_gp(name):
        return None
    try:
        tree = parse(name)
    except GPError as e:
        raise ResolveError(str(e), "E_BAD_EXPR", hint="python -m rsi gp check EXPR") from None
    return functools.partial(ExprActivation, tree, name.startswith("gpl:"))


def install_resolver():
    """Let ACTIVATIONS.get('gp:...') build expression activations (idempotent)."""
    if not _INSTALLED:
        ACTIVATIONS.add_resolver(_resolve)
        _INSTALLED.append(True)


def check(expr):
    """rsi/gp-check@1: parse, simplify and sanity-check one expression."""
    try:
        tree = parse(expr)
    except GPError as e:
        raise RsiError(str(e), "E_BAD_EXPR", value=expr, hint="e.g. gp:mul(x;sin(mul(3;x)))") from None
    simple = simplify(tree)
    ok, why = sanity(simple)
    learn = str(expr).startswith("gpl:")
    with torch.no_grad():
        sample = {str(v): round(float(evaluate(simple, torch.tensor([v]))[0]), 6) for v in (-2.0, -1.0, 0.0, 1.0, 2.0)}
    return {"expr": expr, "name": to_name(tree, learn), "infix": to_infix(tree), "simplified": to_name(simple, learn),
            "simplified_infix": to_infix(simple), "n_nodes": size(tree), "depth": depth(tree), "ok": ok,
            "reason": why, "semantic_hash": semantic_hash(simple) if ok else None, "learnable_consts": learn,
            "sample": sample}


install_resolver()
