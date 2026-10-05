"""
space: the search space of `rsi evolve` (and `sweep --random`): genes, the
genome -> config map (phenotype), and the operators (sample, mutate, crossover).

    sp = Space.load("default", base_cfg, pinned=["model"])   # rsi/space@1 file, dict or "default"
    g = sp.encode(base_cfg)                                   # genome: {gene: value}
    sp.phenotype(g) == config_to_dict(base_cfg)               # individual 0 == a plain run of the base
    child, ops = sp.mutate(sp.crossover(a, b, rng)[0], rng)   # all randomness from one random.Random

A genome is a plain dict (gene name -> JSON value; layers as [[width, act], ...]).
The task (dataset, n_points, noise, splitter, test_frac, seed) is never a gene.
A gene is active when its `when` holds on the phenotype; inactive genes keep the
base value, so genomes that differ only there share a config_key.

Part of nn-playground. AGPL-3.0; for other licensing see COMMERCIAL.md.
"""
import copy
import json
import math
from pathlib import Path

from nncore import configio, schema
from nncore.config import DATA_KEYS, MODEL_KEYS, Config
from nncore.models import parse_layers

from .errors import RsiError

KINDS = ("choice", "part", "activation", "ordinal", "int", "float", "subset", "layers")
DEFAULT_PATH = Path(__file__).resolve().parent / "spaces" / "default.json"
TASK_FIELDS = set(DATA_KEYS)
MODEL_DEP = set(MODEL_KEYS) - schema.ALWAYS_READ  # genes whose effect depends on the model
GROUPS = {"opt": ("optimizer", "lr", "weight_decay", "schedule"), "batch": ("sampler", "batch_size")}
LAYER_OPS = (("width", 0.35), ("act", 0.25), ("insert", 0.15), ("delete", 0.10), ("duplicate", 0.10),
             ("swap", 0.05))
Genome = dict  # gene name -> value


def _bad(msg, **kw):
    return RsiError(msg, "E_BAD_SPACE", hint="python -m rsi schema space", **kw)


def _field(name):
    return name.split(".", 1)[0] if name.startswith("extra.") else name


def poisson(rng, lam):
    """Knuth's method on the GA rng."""
    if lam <= 0:
        return 0
    lim, k, p = math.exp(-lam), 0, rng.random()
    while p > lim:
        k += 1
        p *= rng.random()
    return k


def _same(a, b):
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


class Space:
    """A resolved search space over a base Config. Build it with Space.load()."""

    def __init__(self, doc, base):
        self.doc = doc
        self.base = configio.normalise(base) if isinstance(base, Config) else configio.config_from_dict(base)[0]
        self.base_dict = configio.config_to_dict(self.base)
        self.genes = doc["genes"]
        self.ui_safe = doc.get("ui_safe", True)
        self.weights = doc.get("mutation_weights") or {}
        self.constraints = doc.get("constraints") or {}
        self.gp = None  # {"rate", "max_depth", "max_nodes", "learnable"} when --gp-activations
        self._gp_seen = {}

    # ---------------------------------------------------------------- loading
    @classmethod
    def load(cls, spec, base, *, pinned=(), freeze=(), unfreeze=(), genes=None, models=None):
        """spec: 'default' | path | dict (rsi/space@1, or a resolved space.json). pinned
        fields (set on the command line) and `freeze` are dropped from the genes,
        `unfreeze` brings frozen ones back, `genes` keeps only those, `models` restricts
        the model choices. '*' choices are resolved now against the loaded parts."""
        doc = _read(spec)
        base = configio.normalise(base) if isinstance(base, Config) else configio.config_from_dict(base)[0]
        if doc.get("resolved"):
            return cls(doc, base)
        return cls(_resolve(doc, base, set(pinned), set(freeze or ()), set(unfreeze or ()), genes, models), base)

    def to_dict(self):
        """The resolved space (written to DIR/space.json; Space.load() reads it back as is)."""
        return copy.deepcopy(self.doc)

    # ---------------------------------------------------------------- genome <-> config
    def encode(self, cfg):
        """The genome of a config (its gene fields' values)."""
        cfg = cfg if isinstance(cfg, Config) else configio.config_from_dict(cfg)[0]
        g = {}
        for name, gene in self.genes.items():
            if name.startswith("extra."):  # None: the base has no such key (the part's own default)
                g[name] = copy.deepcopy(cfg.extra.get(name[6:], gene.get("default")))
            elif gene["kind"] == "layers":
                try:
                    g[name] = [[w, a] for w, a in parse_layers(cfg.layers, cfg.activation)]
                except ValueError:
                    g[name] = []
            elif gene["kind"] == "subset":
                g[name] = list(getattr(cfg, name))
            else:
                g[name] = copy.deepcopy(getattr(cfg, name))
        return g

    def active(self, genome):
        """Names of the genes whose `when` holds on the phenotype (a small fixed point)."""
        act = None
        view = dict(self.base_dict)
        for _ in range(len(self.genes) + 1):
            new = [n for n in self.genes if n in genome and self._when(n, view)]
            if new == act:
                break
            act = new
            view = dict(self.base_dict)
            for n in act:
                if not n.startswith("extra.") and self.genes[n]["kind"] not in ("layers", "subset"):
                    view[n] = self._value(n, genome[n])
        return act

    def _when(self, name, view):
        return all(view.get(f) in vals for f, vals in (self.genes[name].get("when") or {}).items())

    def _value(self, name, v):
        """A gene value as its config value (floats snapped unless equal to the base)."""
        gene, f = self.genes[name], _field(name)
        if gene["kind"] == "float":
            base = self.base.extra.get(name[6:]) if name.startswith("extra.") else getattr(self.base, f)
            return base if v == base else _snap(gene, v)
        if gene["kind"] == "int":
            return int(round(v))
        return copy.deepcopy(v)

    def phenotype(self, genome):
        """Base config + every active gene, as a config dict (config_to_dict form)."""
        d = copy.deepcopy(self.base_dict)
        later = []
        for n in self.active(genome):
            gene = self.genes[n]
            if n.startswith("extra."):
                d["extra"] = {k: v for k, v in d["extra"].items() if k != n[6:]}
                if genome[n] is not None:  # None = left out, as in the base
                    d["extra"][n[6:]] = self._value(n, genome[n])
            elif gene["kind"] in ("layers", "subset"):
                later.append(n)
            else:
                d[n] = self._value(n, genome[n])
        for n in later:
            v = genome[n]
            if self.genes[n]["kind"] == "subset":  # registry order, unless it is the base's own set
                d[n] = list(self.base.features) if set(v) == set(self.base.features) else \
                    [x for x in self.genes[n]["choices"] if x in v] + [x for x in v if x not in self.genes[n]["choices"]]
            else:
                d[n] = self._layers_text(v, d["activation"])
        return configio.config_to_dict(configio.normalise(Config(**d)))

    def _layers_text(self, layers, act):
        spec = [(int(w), a) for w, a in layers]
        try:
            if parse_layers(self.base.layers, act) == spec:
                return self.base.layers  # the base string verbatim
        except ValueError:
            pass
        return ",".join(f"{w}:{a}" for w, a in spec)

    def genome_id(self, genome):
        return configio.config_key(self.phenotype(genome))[:12]

    # ---------------------------------------------------------------- validity
    def ui_problems(self, pheno):
        """Gene fields whose phenotype value the UI widgets cannot show exactly (ui_safe)."""
        if not self.ui_safe:
            return []
        cfg = configio.config_from_dict(pheno)[0] if isinstance(pheno, dict) else pheno
        return [i for i in configio.ui_issues(cfg) if i["code"] == "W_NOT_UI_EXACT" and i["field"] in self.genes]

    # ---------------------------------------------------------------- sampling
    def sample(self, rng):
        """A uniformly random genome (log-uniform for log genes)."""
        return {n: self._sample(g, rng) for n, g in self.genes.items()}

    def _sample(self, g, rng):
        k = g["kind"]
        if k in ("choice", "part", "activation", "ordinal"):
            return copy.deepcopy(rng.choice(g["choices"]))
        if k == "int":
            if g.get("log"):
                return int(round(math.exp(rng.uniform(math.log(g["low"]), math.log(g["high"])))))
            return rng.randint(int(g["low"]), int(g["high"]))
        if k == "float":
            if g.get("zero_prob") and rng.random() < g["zero_prob"]:
                return 0.0
            lo, hi = g["low"], g["high"]
            v = math.exp(rng.uniform(math.log(lo), math.log(hi))) if g.get("log") else rng.uniform(lo, hi)
            return _snap(g, v)
        if k == "subset":
            must = g.get("must_include") or []
            v = [x for x in g["choices"] if x in must or rng.random() < 0.3]
            while len(v) < g.get("min", 1):
                v.append(rng.choice([x for x in g["choices"] if x not in v]))
            return [x for x in g["choices"] if x in v]
        if k == "layers":
            d = rng.randint(g["min_depth"], g["max_depth"])
            return [[rng.choice(g["widths"]), rng.choice(g["act_choices"])] for _ in range(d)]
        raise _bad(f"unknown gene kind {k!r}")

    # ---------------------------------------------------------------- mutation
    def mutable(self, genome):
        """Active genes that can take another value."""
        return [n for n in self.active(genome) if _domain(self.genes[n], genome[n]) > 1]

    def mutate(self, genome, rng, n_ops=None, mutations=1.5):
        """(child, ops): n = 1 + Poisson(mutations - 1) operations (or n_ops), each on one
        active gene picked by mutation_weights; every operation changes its gene."""
        child = copy.deepcopy(genome)
        n = n_ops if n_ops is not None else 1 + poisson(rng, mutations - 1)
        ops, i = [], 0
        while i < n or (i < n + 10 and _same(child, genome)):  # two ops may undo each other
            i += 1
            cands = self.mutable(child)
            if not cands:
                break
            w = [float(self.weights.get(c, 1.0)) for c in cands]
            name = rng.choices(cands, weights=w)[0]
            child[name] = self._mutate(name, child[name], rng)
            ops.append(f"mutate:{name}")
        return child, ops

    def _mutate(self, name, v, rng):
        g = self.genes[name]
        k = g["kind"]
        if v is None and name.startswith("extra."):
            return self._sample(g, rng)  # the base had no value: start from a sample
        if k == "activation" and self.gp is not None:
            out = self._mutate_gp(v, g, rng)
            if out is not None and out != v:
                return out
        if k in ("choice", "part", "activation"):
            return copy.deepcopy(rng.choice([c for c in g["choices"] if not _same(c, v)]))
        if k == "ordinal":
            ch = g["choices"]
            i = _index(ch, v)
            step = 1 if rng.random() < 0.8 else 2
            j = _reflect(i + rng.choice((-1, 1)) * step, len(ch) - 1)
            if j == i:
                j = i + 1 if i + 1 < len(ch) else i - 1
            return copy.deepcopy(ch[j])
        if k == "int":
            lo, hi = int(g["low"]), int(g["high"])
            for _ in range(10):
                if g.get("log") and v > 0:
                    new = int(round(v * math.exp(rng.gauss(0, 0.35))))
                else:
                    new = v + int(round(rng.gauss(0, max(1.0, 0.15 * (hi - lo)))))
                new = min(hi, max(lo, new))
                if new != v:
                    return new
            return v + 1 if v + 1 <= hi else v - 1
        if k == "float":
            return self._mutate_float(g, v, rng)
        if k == "subset":
            must, cur = set(g.get("must_include") or []), list(v)
            flips = [x for x in g["choices"] if x not in cur] + \
                [x for x in cur if x not in must and len(cur) - 1 >= g.get("min", 1)]
            x = rng.choice(flips)
            new = [y for y in cur if y != x] if x in cur else cur + [x]
            return [y for y in g["choices"] if y in new]
        if k == "layers":
            return self._mutate_layers(g, v, rng)
        raise _bad(f"unknown gene kind {k!r}")

    def _mutate_float(self, g, v, rng):
        lo, hi, zp = g["low"], g["high"], g.get("zero_prob") or 0.0
        for _ in range(10):
            if zp and rng.random() < zp:
                new = 0.0 if v != 0 else self._sample({**g, "zero_prob": 0}, rng)
            elif rng.random() < 0.1 or (g.get("log") and v <= 0):
                new = self._sample({**g, "zero_prob": 0}, rng)
            elif g.get("log"):
                x = math.log(v) + rng.gauss(0, 0.25 * math.log(10))  # sigma = 0.25 decades
                new = math.exp(_reflect_f(x, math.log(lo), math.log(hi)))
            else:
                new = _reflect_f(v + rng.gauss(0, 0.15 * (hi - lo)), lo, hi)
            new = _snap(g, new)
            if new != v:
                return new
        return _snap(g, hi if v < hi else lo)

    def _mutate_layers(self, g, v, rng):
        layers, d = [list(x) for x in v], len(v)
        lo, hi, W, A = g["min_depth"], g["max_depth"], g["widths"], g["act_choices"]
        ok = {"width": d >= 1 and len(W) > 1, "act": d >= 1 and len(A) > 1, "insert": d < hi,
              "delete": d > lo, "duplicate": 1 <= d < hi,
              "swap": d >= 2 and any(layers[i] != layers[i + 1] for i in range(d - 1))}
        ops = [(o, p) for o, p in LAYER_OPS if ok[o]]
        op = rng.choices([o for o, _ in ops], weights=[p for _, p in ops])[0]
        if op == "width":
            i = rng.randrange(d)
            j = _index(W, layers[i][0])
            layers[i][0] = W[_reflect(j + rng.choice((-1, 1)), len(W) - 1)]
        elif op == "act":
            i = rng.randrange(d)
            layers[i][1] = rng.choice([a for a in A if a != layers[i][1]])
        elif op == "insert":
            pos = rng.randint(0, d)
            src = layers[pos - 1] if pos > 0 else (layers[0] if d else [rng.choice(W), rng.choice(A)])
            layers.insert(pos, list(src))
        elif op == "delete":
            del layers[rng.randrange(d)]
        elif op == "duplicate":  # repeat the stack (block duplication), clipped to max_depth
            layers = (layers + [list(x) for x in layers])[:hi]
        else:
            i = rng.choice([i for i in range(d - 1) if layers[i] != layers[i + 1]])
            layers[i], layers[i + 1] = layers[i + 1], layers[i]
        return layers

    def _mutate_gp(self, v, g, rng):
        from . import gp
        if rng.random() >= self.gp["rate"]:
            return None  # plain registry resample
        if isinstance(v, str) and gp.is_gp(v):
            tree = gp.mutate(gp.parse(v), rng, self.gp["max_depth"], self.gp["max_nodes"])
        else:
            tree = gp.Node(rng.choice(gp.UNARY), (gp.Node("x"),))
        return self._gp_name(tree)

    def _gp_name(self, tree):
        """Simplified, sane and semantically deduplicated GP name, or None."""
        from . import gp
        tree = gp.simplify(tree)
        ok, _ = gp.sanity(tree)
        if not ok:
            return None
        h = gp.semantic_hash(tree)
        name = self._gp_seen.setdefault(h, gp.to_name(tree, learnable=self.gp.get("learnable", False)))
        return name

    # ---------------------------------------------------------------- crossover
    def arch_genes(self):
        return [n for n in self.genes if n == "model" or n in MODEL_DEP or n.startswith("extra.")]

    def crossover(self, a, b, rng):
        """(child, ops). Linkage groups, each inherited by a coin flip: arch (model + model
        dependent genes + extra.*; whole from one parent when the models differ, else uniform
        with one-point layers crossover), opt, batch, then every other gene on its own."""
        child, ops = {}, []
        arch = self.arch_genes()
        if arch:
            if "model" in arch and a.get("model") != b.get("model"):
                src = a if rng.random() < 0.5 else b
                for n in arch:
                    child[n] = copy.deepcopy(src[n])
                ops.append("crossover:arch")
            else:
                for n in arch:
                    child[n] = self._cross_gene(n, a[n], b[n], rng)
                ops.append("crossover:arch-uniform")
        for grp, names in GROUPS.items():
            names = [n for n in names if n in self.genes]
            if names:
                src = a if rng.random() < 0.5 else b
                for n in names:
                    child[n] = copy.deepcopy(src[n])
                ops.append(f"crossover:{grp}")
        for n in self.genes:
            if n not in child:
                child[n] = copy.deepcopy(a[n] if rng.random() < 0.5 else b[n])
        return {n: child[n] for n in self.genes}, ops

    def _cross_gene(self, n, x, y, rng):
        g = self.genes[n]
        if g["kind"] == "layers":
            for _ in range(10):
                i, j = rng.randint(0, len(x)), rng.randint(0, len(y))
                out = ([list(v) for v in x[:i]] + [list(v) for v in y[j:]])[:g["max_depth"]]
                if len(out) >= g["min_depth"]:
                    return out
            return copy.deepcopy(x)
        if g["kind"] == "activation" and self.gp is not None:
            from . import gp
            if isinstance(x, str) and isinstance(y, str) and gp.is_gp(x) and gp.is_gp(y):
                name = self._gp_name(gp.crossover(gp.parse(x), gp.parse(y), rng, self.gp["max_depth"],
                                                  self.gp["max_nodes"]))
                if name is not None:
                    return name
        return copy.deepcopy(x if rng.random() < 0.5 else y)

    # ---------------------------------------------------------------- distances
    def distance(self, a, b):
        """Normalised Hamming distance over the genes active in either genome."""
        act = set(self.active(a)) | set(self.active(b))
        if not act:
            return 0.0
        return sum(not _same(a.get(n), b.get(n)) for n in act) / len(act)


# ---------------------------------------------------------------- helpers
def _index(ch, v):
    for i, c in enumerate(ch):
        if _same(c, v):
            return i
    nums = [(abs(c - v), i) for i, c in enumerate(ch) if isinstance(c, (int, float)) and isinstance(v, (int, float))]
    return min(nums)[1] if nums else 0


def _reflect(i, top):
    if top <= 0:
        return 0
    while i < 0 or i > top:
        i = -i if i < 0 else 2 * top - i
    return i


def _reflect_f(x, lo, hi):
    for _ in range(8):
        if x < lo:
            x = 2 * lo - x
        elif x > hi:
            x = 2 * hi - x
        else:
            return x
    return min(hi, max(lo, x))


def _snap(g, v):
    """quantum snapping, else `sig` significant digits (default 4); kept in [low, high]."""
    if v == 0:
        return 0.0
    lo, hi = g.get("low", -math.inf), g.get("high", math.inf)
    v = min(hi, max(lo, float(v)))
    if g.get("quantum"):
        q = g["quantum"]
        v = round(round(v / q) * q, 12)
        if v < lo - 1e-12:
            v = round(v + q, 12)
        if v > hi + 1e-12:
            v = round(v - q, 12)
        return v
    return float(f"{v:.{int(g.get('sig', 4))}g}")


def _domain(g, v):
    k = g["kind"]
    if k in ("choice", "part", "activation", "ordinal"):
        n = len(g["choices"])
        return n + (0 if any(_same(c, v) for c in g["choices"]) else 1)
    if k in ("int", "float"):
        return 2 if g["high"] > g["low"] else 1
    if k == "subset":
        return 2 if any(x not in (g.get("must_include") or []) for x in g["choices"]) else 1
    if k == "layers":
        return 2 if g["max_depth"] > 0 and (g["max_depth"] > g["min_depth"] or len(g["widths"]) > 1
                                            or len(g["act_choices"]) > 1) else 1
    return 1


def _read(spec):
    if spec is None or spec == "default":
        spec = DEFAULT_PATH
    if isinstance(spec, dict):
        doc = copy.deepcopy(spec)
    else:
        p = Path(spec)
        if not p.is_file():
            alt = DEFAULT_PATH.parent / f"{spec}.json"
            if not alt.is_file():
                raise RsiError(f"space file not found: {spec}", "E_NOT_FOUND", value=str(spec))
            p = alt
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except ValueError as e:
            raise _bad(f"space file {p} is not valid JSON: {e}") from None
    if not isinstance(doc, dict) or doc.get("format") != "rsi/space" or doc.get("version") != 1:
        raise _bad("not an rsi/space@1 document (need format 'rsi/space', version 1)")
    if not isinstance(doc.get("genes"), dict):
        raise _bad("space needs a 'genes' object")
    return doc


def _names(field, registry=None):
    reg = registry or configio.FIELD_REGISTRY.get(field)
    if reg not in configio.PART_REGISTRIES:
        raise _bad(f"gene {field}: unknown registry {reg!r}", field=field,
                   allowed=sorted(configio.PART_REGISTRIES))
    return list(configio.PART_REGISTRIES[reg])


def _canon(field, v):
    if field.startswith("extra."):
        return v
    try:
        v = configio.coerce_value(field, v)
        if field in configio.FIELD_REGISTRY and field != "features" and \
                v not in configio.SPECIAL_VALUES.get(field, ()):
            v = configio.resolve_name(field, v)[0]
    except Exception as e:
        raise _bad(f"gene {field}: bad value {v!r}: {e}", field=field, value=v) from None
    return v


def _resolve(doc, base, pinned, freeze, unfreeze, keep, models):
    raw = dict(doc["genes"])
    for f in unfreeze:
        if f not in raw:
            if f not in configio.FIELD_REGISTRY or f in TASK_FIELDS:
                raise _bad(f"--unfreeze {f}: no gene for it in the space; add one to the space file", field=f)
            raw[f] = {"kind": "part", "choices": "*"}
    drop = (set(doc.get("frozen_by_default") or ()) | pinned | freeze) - unfreeze
    if keep:
        missing = [g for g in keep if g not in raw]
        if missing:
            raise _bad(f"--genes: {missing[0]!r} is not a gene of this space", field=missing[0], allowed=list(raw))
        drop |= set(raw) - set(keep)
    ui_safe = doc.get("ui_safe", True)
    genes, dropped = {}, {}
    for name, g in raw.items():
        f = _field(name)
        if f not in configio.FIELDS or f == "extra" and not name.startswith("extra."):
            raise _bad(f"gene {name!r} is not a Config field (or extra.NAME)", field=name)
        if f in TASK_FIELDS:
            raise _bad(f"gene {name!r} is part of the task (dataset, n_points, noise, splitter, test_frac, seed) "
                       "and is never searched; use --datasets / --suite", field=name)
        if not isinstance(g, dict) or g.get("kind") not in KINDS:
            raise _bad(f"gene {name!r}: kind must be one of {', '.join(KINDS)}", field=name)
        if name in drop:
            dropped[name] = "pinned" if name in pinned else "frozen"
            continue
        genes[name] = _gene(name, g, ui_safe)
    if models and "model" in genes:
        ms = [_canon("model", m) for m in models]
        genes["model"]["choices"] = [m for m in genes["model"]["choices"] if m in ms] or ms
    elif models and "model" not in genes and len(models) > 1:
        genes["model"] = _gene("model", {"kind": "choice", "choices": list(models)}, ui_safe)
    mods = genes["model"]["choices"] if "model" in genes else [base.model]
    inert = doc.get("inert") or {}
    for name, g in list(genes.items()):
        if name in MODEL_DEP:
            readers = [m for m in mods if name in schema.fields_read(m) and name not in inert.get(m, ())]
            if not readers:
                dropped[name] = "not read by " + ", ".join(mods)
                del genes[name]
                continue
            w = dict(g.get("when") or {})
            g["when"] = w if "model" in w else {"model": readers, **w}
        if g["kind"] in ("choice", "part", "activation", "ordinal") and len(g["choices"]) <= 1 or \
                g["kind"] in ("int", "float") and g["high"] <= g["low"]:
            dropped[name] = "single value"
            del genes[name]
    out = {k: v for k, v in doc.items() if k != "genes"}
    out.update(resolved=True, ui_safe=ui_safe, genes=genes, dropped=dropped,
               constraints={"max_params": 20000, "max_est_seconds": 120, **(doc.get("constraints") or {})})
    return out


def _feature(c):
    try:
        return configio.resolve_name("features", c)[0]
    except Exception as e:
        raise _bad(f"gene features: bad value {c!r}: {e}", field="features", value=c) from None


def _gene(name, g, ui_safe):
    g = copy.deepcopy(g)
    k, f = g["kind"], _field(name)
    bounds = configio.FIELD_BOUNDS.get(f)
    ui = configio.UI_LIMITS.get(f) if ui_safe else None
    if k in ("part", "activation"):
        reg = g.get("registry") or ("activation" if k == "activation" else configio.FIELD_REGISTRY.get(f))
        g["registry"] = reg
        ch = g.get("choices", "*")
        names = _names(f, reg)
        ch = [n for n in names if n not in (g.get("exclude") or ())] if ch == "*" else [_canon(name, c) for c in ch]
        for c in ch:
            if c not in names:
                raise _bad(f"gene {name}: unknown {reg} {c!r}", field=name, value=c)
        ch += [c for c in (g.get("extra_choices") or ()) if c not in ch]
        g["choices"] = ch
    elif k in ("choice", "ordinal"):
        if not isinstance(g.get("choices"), list) or not g["choices"]:
            raise _bad(f"gene {name}: {k} needs a non-empty choices list", field=name)
        g["choices"] = [_canon(name, c) for c in g["choices"]]
    elif k in ("int", "float"):
        if not all(isinstance(g.get(x), (int, float)) for x in ("low", "high")) or g["low"] > g["high"]:
            raise _bad(f"gene {name}: {k} needs numbers low <= high", field=name)
        if g.get("log") and g["low"] <= 0:
            raise _bad(f"gene {name}: log scale needs low > 0", field=name)
        if bounds and not (bounds[0] <= g["low"] and g["high"] <= bounds[1]):
            raise _bad(f"gene {name}: [{g['low']}, {g['high']}] is outside the field bounds {list(bounds)}",
                       field=name)
        if isinstance(ui, tuple) and not (ui[0] <= g["low"] and g["high"] <= ui[1]):
            raise _bad(f"gene {name}: [{g['low']}, {g['high']}] is outside the UI range {list(ui)} (ui_safe)",
                       field=name)
    elif k == "subset":
        names = _names(f, g.get("registry"))
        ch = g.get("choices", "*")
        g["choices"] = names if ch == "*" else [_feature(c) for c in ch]
        g["must_include"] = [_feature(c) for c in (g.get("must_include") or [])]
        g["min"] = max(1, int(g.get("min", 1)), len(g["must_include"]))
    elif k == "layers":
        if f != "layers":
            raise _bad(f"gene {name}: kind layers only fits the 'layers' field", field=name)
        g["min_depth"], g["max_depth"] = int(g.get("min_depth", 0)), int(g.get("max_depth", 4))
        g["widths"] = [int(w) for w in g.get("widths") or [2, 4, 8, 16, 32]]
        if not 0 <= g["min_depth"] <= g["max_depth"] or any(not 1 <= w <= 512 for w in g["widths"]):
            raise _bad(f"gene {name}: need 0 <= min_depth <= max_depth and widths in 1..512", field=name)
        act = g.get("act") or {}
        names = _names("activation")
        ch = act.get("choices", "*")
        g["act_choices"] = [n for n in names if n not in (act.get("exclude") or ())] if ch == "*" else \
            [_canon("activation", c) for c in ch]
    if k in ("choice", "ordinal", "part"):
        for c in g["choices"]:
            if bounds and isinstance(c, (int, float)) and not bounds[0] <= c <= bounds[1]:
                raise _bad(f"gene {name}: value {c!r} is outside the field bounds {list(bounds)}", field=name, value=c)
            if isinstance(ui, tuple) and isinstance(c, (int, float)) and not ui[0] <= c <= ui[1]:
                raise _bad(f"gene {name}: value {c!r} is outside the UI range {list(ui)} (ui_safe)", field=name, value=c)
            if isinstance(ui, dict) and "choices" in ui and c not in ui["choices"]:
                raise _bad(f"gene {name}: value {c!r} is not offered by the UI {ui['choices']} (ui_safe)", field=name,
                           value=c)
    for f2, vals in (g.get("when") or {}).items():
        if f2 not in configio.FIELDS or not isinstance(vals, list):
            raise _bad(f"gene {name}: when needs {{field: [values]}}", field=name)
    return g
