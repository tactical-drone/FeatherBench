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


def check_submission(path, github=None):
    """(Config, github) for a submission file, or RsiError. No code runs here."""
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
    try:
        cfg = configio.load_settings(doc, strict=True).config  # unknown fields / parts / values raise
    except configio.ConfigError as e:
        raise _bad(f"{path.name}: {e}", field=getattr(e, "field", None)) from None
    return cfg, user


@closes_stores
def verify(path, *, github=None, benchmark=None, workers=None, out=None, store=None, on_event=None):
    """Check a submission, then run the official benchmark on it. Returns the rsi/bench doc
    plus "github" (and writes it to `out`)."""
    from . import api
    from .bench import DEFAULT, get_benchmark
    from nncore import schema
    api.setup_parts(None)  # the repo's own parts only
    cfg, user = check_submission(path, github)
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
                        "code_fp": e["code_fp"]})
    board = {"format": "featherbench/leaderboard", "version": 1, "name": "FeatherBench",
             "rule": "fewest params that solves every pattern wins",
             "benchmark": {"id": b.id, "hash": b.hash}, "n_entries": len(entries),
             "entries": entries[:int(top)] if top else entries}
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        configio.atomic_write_text(out, json.dumps(configio.json_safe(board), ensure_ascii=False, indent=1) + "\n")
    return board
