"""
featherbench: community attempts by pull request, checked by CI.

    python -m rsi featherbench verify featherbench/submissions/USER.json --github USER [--out result.json]
    python -m rsi featherbench leaderboard [--results featherbench/results] [--out featherbench/leaderboard.json]

A submission is DATA ONLY: one JSON settings file (what File > Save settings or
`rsi export` writes, or a bare config object) at featherbench/submissions/<github user>.json.
It never names code: only parts that ship with this repo (my_parts) are loaded, and any
"parts" / meta.parts entry other than that is refused.

verify checks the file (size, shape, name, known fields and parts, a params cap) and then
runs the official benchmark on it. CI runs it on every pull request with a read-only token;
the leaderboard numbers come from the run on main after merge (featherbench/results/), so
nothing a pull request computes is ever trusted.

Part of nn-playground. AGPL-3.0; for other licensing see COMMERCIAL.md.
"""
import json
import re
from pathlib import Path

from nncore import configio

from .errors import RsiError
from .store import closes_stores

SUBMISSIONS = Path("featherbench") / "submissions"
RESULTS = Path("featherbench") / "results"
LEADERBOARD = Path("featherbench") / "leaderboard.json"
MAX_BYTES = 64 * 1024
MAX_PARAMS = 20000  # FeatherBench is about small nets; also bounds what a submission costs CI
GITHUB_USER = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?$")
SETTINGS_KEYS = {"format", "version", "config", "train", "ui", "meta", "parts"}
ALLOWED_PARTS = {"my_parts"}


def _bad(msg, **kw):
    return RsiError(msg, "E_BAD_SETTINGS", **kw)


# meta.attempt: who (or what) found the setup and what it took. Self-reported, shown on the
# leaderboard as such; CI can't check it. Every field is optional; "unknown" is honest.
ATTEMPT_FIELDS = {
    "model": "the AI model that drove the attempt, as its maker names it (or 'human')",
    "harness": "the agent / tool it ran in, e.g. 'Claude Code', 'Codex CLI', 'Cursor', 'manual'",
    "tokens": "total tokens used for the attempt (input + output), an integer",
    "cost_usd": "what the attempt cost in US dollars (model usage; compute you paid for)",
    "human_assist": "'none' (the AI did it alone), 'some' or 'lots'",
}
HUMAN_ASSIST = ("none", "some", "lots")


def check_attempt(attempt, name="submission"):
    """A cleaned meta.attempt dict (unknown values -> None), or RsiError."""
    if attempt is None:
        return None
    if not isinstance(attempt, dict):
        raise _bad(f"{name}: meta.attempt must be an object with {sorted(ATTEMPT_FIELDS)}")
    extra = sorted(set(attempt) - set(ATTEMPT_FIELDS))
    if extra:
        raise _bad(f"{name}: meta.attempt has unknown keys {extra}; allowed: {sorted(ATTEMPT_FIELDS)}", value=extra)
    out = {}
    for k in ATTEMPT_FIELDS:
        v = attempt.get(k)
        if v is None or (isinstance(v, str) and v.strip().lower() in ("", "unknown", "n/a")):
            out[k] = None
            continue
        if k in ("model", "harness"):
            if not isinstance(v, str) or len(v) > 80 or not v.isprintable():
                raise _bad(f"{name}: meta.attempt.{k} must be a short printable string (<= 80 chars)", field=k)
            out[k] = v.strip()
        elif k == "tokens":
            if isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= 10 ** 11:
                raise _bad(f"{name}: meta.attempt.tokens must be a whole number of tokens", field=k, value=v)
            out[k] = v
        elif k == "cost_usd":
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 10 ** 6:
                raise _bad(f"{name}: meta.attempt.cost_usd must be a number of US dollars", field=k, value=v)
            out[k] = round(float(v), 4)
        elif k == "human_assist":
            if v not in HUMAN_ASSIST:
                raise _bad(f"{name}: meta.attempt.human_assist must be one of {list(HUMAN_ASSIST)}", field=k, value=v)
            out[k] = v
    return out


def stamp(path, **fields):
    """Write meta.attempt into a settings file (other content kept). Returns the attempt."""
    path = Path(path)
    doc = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(doc, dict):
        raise _bad(f"{path.name} must be a JSON object")
    if "format" not in doc and "config" not in doc:  # a bare config: wrap it as a settings doc
        doc = {"format": configio.SETTINGS_FORMAT, "version": configio.SETTINGS_VERSION, "config": doc}
    meta = doc.setdefault("meta", {})
    merged = {**(meta.get("attempt") or {}), **{k: v for k, v in fields.items() if v is not None}}
    meta["attempt"] = check_attempt(merged, path.name)
    configio.atomic_write_text(path, json.dumps(doc, ensure_ascii=False, indent=1) + "\n")
    return meta["attempt"]


def feather_score(summary):
    """0..1000, ordered exactly like the ranking: unsolved setups score below 923 (12/13 of
    the scale, by patterns solved and mean accuracy); solving all 12 scores 923..1000, more
    for fewer params (log scale). A scale to plot per model over time."""
    from .bench import scalar
    n = summary.get("n_datasets") or 12
    return round(1000 * scalar(summary) / (n + 1), 1)


def check_submission(path, github=None):
    """(Config, github, attempt) for a submission file, or RsiError. No code runs here."""
    path = Path(path)
    user = path.stem
    if path.suffix != ".json":
        raise _bad(f"{path.name}: a submission is one .json file", value=str(path))
    if not GITHUB_USER.match(user):
        raise _bad(f"{path.name}: name the file after your GitHub username (<user>.json)", value=user)
    if github is not None and user.lower() != str(github).lower():
        raise _bad(f"{path.name} belongs to '{user}', but the pull request is from '{github}': "
                   f"submit featherbench/submissions/{github}.json", value=user)
    try:
        raw = path.read_bytes()
    except OSError as e:
        raise RsiError(f"cannot read {path}: {e}", "E_NOT_FOUND", value=str(path)) from None
    if len(raw) > MAX_BYTES:
        raise _bad(f"{path.name} is {len(raw)} bytes; the limit is {MAX_BYTES}", value=len(raw))
    try:
        doc = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, ValueError) as e:
        raise _bad(f"{path.name} is not valid UTF-8 JSON: {e}") from None
    if not isinstance(doc, dict):
        raise _bad(f"{path.name} must be a JSON object")
    attempt = None
    if "format" in doc or "config" in doc:  # a settings doc
        extra = sorted(set(doc) - SETTINGS_KEYS)
        if extra:
            raise _bad(f"{path.name}: unexpected keys {extra} (a settings doc has {sorted(SETTINGS_KEYS)})",
                       value=extra)
        named = list((doc.get("parts") or {}).keys()) if isinstance(doc.get("parts"), dict) else \
            list(doc.get("parts") or [])
        named += list((doc.get("meta") or {}).get("parts") or [])
        foreign = sorted({str(p) for p in named} - ALLOWED_PARTS)
        if foreign:
            raise _bad(f"{path.name} names part modules {foreign}: submissions are data only, built from the "
                       f"parts in this repo ({sorted(ALLOWED_PARTS)})", value=foreign)
        attempt = check_attempt((doc.get("meta") or {}).get("attempt"), path.name)
    try:
        cfg = configio.load_settings(doc, strict=True).config  # unknown fields / parts / values raise
    except configio.ConfigError as e:
        raise _bad(f"{path.name}: {e}", field=getattr(e, "field", None)) from None
    return cfg, user, attempt


@closes_stores
def verify(path, *, github=None, benchmark=None, workers=None, out=None, store=None, on_event=None):
    """Check a submission, then run the official benchmark on it. Returns the rsi/bench doc
    plus "github" (and writes it to `out`)."""
    from . import api
    from .bench import DEFAULT, get_benchmark
    from nncore import schema
    api.setup_parts(None)  # the repo's own parts only
    cfg, user, attempt = check_submission(path, github)
    b = get_benchmark(benchmark or DEFAULT)
    biggest = 0
    for d in b.datasets:  # params depend on the dataset's class count
        n = schema.count_params(configio.config_from_dict({**configio.config_to_dict(cfg), **b.data(d, b.seeds[0])})[0])
        biggest = max(biggest, n)
        if n > MAX_PARAMS:
            raise _bad(f"{Path(path).name}: {n} params on {d}; FeatherBench submissions are capped at "
                       f"{MAX_PARAMS}", value=n)
    doc = api.bench(cfg, b.id, workers, None, cache=False, store=store, on_event=on_event)
    doc["github"] = user
    doc["attempt"] = attempt  # self-reported (model, harness, tokens, cost_usd, human_assist)
    doc["feather_score"] = feather_score(doc)
    doc["submission_file"] = Path(path).as_posix()
    if out:
        configio.atomic_write_text(out, json.dumps(configio.json_safe(doc), ensure_ascii=False, indent=1) + "\n")
    return doc


def leaderboard(results=RESULTS, out=LEADERBOARD, *, benchmark=None, top=None):
    """Rank the CI results (one rsi/bench doc per racer) into featherbench/leaderboard.json:
    solved_all first, then fewest params_max, then mean accuracy."""
    from .bench import DEFAULT, canonical_id, get_benchmark, rank
    b = get_benchmark(benchmark or DEFAULT)
    files = sorted(Path(results).glob("*.json")) if Path(results).is_dir() else []
    docs = {}
    for f in files:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        bm = d.get("benchmark") or {}
        if canonical_id(bm.get("id")) == b.id and bm.get("hash") == b.hash and d.get("github"):
            docs[str(f)] = d
    ranked = rank(list(docs), top=None)["entries"] if docs else []
    entries = []
    for e in ranked:
        d = docs[e["file"]]
        if not e["consistent"]:
            continue
        entries.append({"rank": len(entries) + 1, "github": d["github"], "params_max": e["params_max"],
                        "mean_acc": e["mean_acc"], "min_acc": e["min_acc"], "solved_all": e["solved_all"],
                        "n_solved": e["n_solved"], "n_datasets": e["n_datasets"], "describe": e["describe"],
                        "submitted": d.get("submitted"), "config_diff": d.get("config_diff"),
                        "config": d.get("config"), "platform": e["platform"], "torch": e["torch"],
                        "code_fp": e["code_fp"], "feather_score": feather_score(e),
                        "attempt": d.get("attempt") or {k: None for k in ATTEMPT_FIELDS}})
    by_model = {}  # the AI view: each model's best entry, and what its attempts cost
    for e in entries:
        m = (e["attempt"] or {}).get("model") or "unknown"
        b_ = by_model.setdefault(m, {"model": m, "best_rank": e["rank"], "best_github": e["github"],
                                     "best_feather_score": e["feather_score"], "best_params_max": e["params_max"],
                                     "best_solved": e["n_solved"], "attempts": 0, "tokens": [], "cost_usd": []})
        b_["attempts"] += 1
        for k in ("tokens", "cost_usd"):
            if (e["attempt"] or {}).get(k) is not None:
                b_[k].append(e["attempt"][k])
    models = []
    for b_ in sorted(by_model.values(), key=lambda x: x["best_rank"]):
        tok, cost = b_.pop("tokens"), b_.pop("cost_usd")
        models.append({**b_, "median_tokens": sorted(tok)[len(tok) // 2] if tok else None,
                       "median_cost_usd": sorted(cost)[len(cost) // 2] if cost else None})
    board = {"format": "featherbench/leaderboard", "version": 1, "name": "FeatherBench",
             "rule": "fewest params that solves every pattern wins",
             "benchmark": {"id": b.id, "hash": b.hash}, "n_entries": len(entries),
             "note": "attempt fields (model, harness, tokens, cost_usd, human_assist) are self-reported; "
                     "scores are CI's own runs",
             "entries": entries[:int(top)] if top else entries, "models": models}
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        configio.atomic_write_text(out, json.dumps(configio.json_safe(board), ensure_ascii=False, indent=1) + "\n")
    return board


# ---------------------------------------------------------------- one-shot (no tools, no repo)
_FENCE = re.compile(r"```(?:json|JSON)?\s*\n(.*?)```", re.S)


def extract_answer(text):
    """The JSON setup in a model's reply: the last ```json fenced block, else the last
    balanced {...} object that parses. None when there is none."""
    for block in reversed(_FENCE.findall(text)):
        try:
            obj = json.loads(block)
            if isinstance(obj, dict):
                return obj
        except ValueError:
            pass
    ends = [i for i, ch in enumerate(text) if ch == "}"]
    for end in reversed(ends):
        depth = 0
        for start in range(end, -1, -1):
            depth += {"}": 1, "{": -1}.get(text[start], 0)
            if depth == 0:
                try:
                    obj = json.loads(text[start:end + 1])
                    if isinstance(obj, dict):
                        return obj
                except ValueError:
                    pass
                break
    return None


def tile_png(cfg, benchmark, path, size=160):
    """Seed-0 decision maps of cfg on every benchmark dataset, tiled into one PNG."""
    import numpy as np
    from nncore import Session
    from .render import map_rgb, write_png
    imgs = []
    for d in benchmark.datasets:
        s = Session(configio.config_from_dict({**configio.config_to_dict(cfg), **benchmark.data(d, benchmark.seeds[0])})[0])
        s.train(benchmark.steps)
        imgs.append(map_rgb(s, size))
    cols, gap = 4, 6
    rows = -(-len(imgs) // cols)
    out = np.full((rows * size + (rows + 1) * gap, cols * size + (cols + 1) * gap, 3), (41, 40, 60), np.uint8)
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        y, x = gap + r * (size + gap), gap + c * (size + gap)
        out[y:y + size, x:x + size] = im[:size, :size]
    write_png(path, out)
    return [str(d) for d in benchmark.datasets]


@closes_stores
def score_answer(source, *, benchmark=None, workers=None, png=None, store=None, on_event=None):
    """Score a one-shot answer (a model's raw reply, as a file path or text). Never raises
    on a bad answer: an answer with no valid setup scores 0 with the reason."""
    from . import api
    from .bench import DEFAULT, get_benchmark
    from nncore import schema
    api.setup_parts(None)
    b = get_benchmark(benchmark or DEFAULT)
    text = Path(source).read_text(encoding="utf-8-sig") if Path(str(source)).is_file() else str(source)
    out = {"format": "featherbench/oneshot", "version": 1, "benchmark": {"id": b.id, "hash": b.hash},
           "valid": False, "feather_score": 0.0, "reason": None, "answer": None}
    obj = extract_answer(text)
    if obj is None:
        out["reason"] = "no JSON object found in the answer"
        return out
    out["answer"] = obj
    tmp = Path(api.open_store_root()) / "oneshot"
    tmp.mkdir(parents=True, exist_ok=True)
    f = tmp / "answer.json"
    f.write_text(json.dumps(obj), encoding="utf-8")
    try:
        cfg, _, _ = check_submission(f)
        for d in b.datasets:
            n = schema.count_params(configio.config_from_dict({**configio.config_to_dict(cfg), **b.data(d, b.seeds[0])})[0])
            if n > MAX_PARAMS:
                raise _bad(f"{n} params on {d}; the cap is {MAX_PARAMS}", value=n)
    except RsiError as e:
        out["reason"] = str(e).replace("answer.json: ", "").replace("answer.json ", "")
        return out
    doc = api.bench(cfg, b.id, workers, None, store=store, on_event=on_event)
    out.update(valid=True, feather_score=feather_score(doc), config_diff=doc["config_diff"],
               **{k: doc[k] for k in ("n_solved", "n_datasets", "solved_all", "params_max", "mean_acc", "min_acc",
                                      "per_dataset", "describe")})
    if png:
        out["png"] = {"file": str(png), "order": tile_png(cfg, b, png)}
    return out
