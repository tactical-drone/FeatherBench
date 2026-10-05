"""
pareto: NSGA-II pieces for `evolve --objective pareto` (maximise accuracy, minimise
parameters) and the accuracy-vs-params front shown on leaderboards.

    fronts = non_dominated_sort(points)             # points: [(acc, -log2 params), ...] (maximised)
    dist = crowding_distance(points, fronts[0])     # {index: distance}, ends = inf
    keep = nsga2_select(points, ids, mu)            # indices of the mu survivors (fronts, then crowding)
    front(rows)                                     # [{genome_id, acc_agg, n_params}] fewest params first

Part of nn-playground. AGPL-3.0; for other licensing see COMMERCIAL.md.
"""
import math


def dominates(a, b):
    """a dominates b: >= on every objective (maximised) and > on one."""
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


def non_dominated_sort(points):
    """Fronts of indices, best first (fast non-dominated sort, Deb et al. 2002)."""
    n = len(points)
    dominated = [[] for _ in range(n)]
    count = [0] * n
    fronts = [[]]
    for i in range(n):
        for j in range(n):
            if i != j:
                if dominates(points[i], points[j]):
                    dominated[i].append(j)
                elif dominates(points[j], points[i]):
                    count[i] += 1
        if count[i] == 0:
            fronts[0].append(i)
    while fronts[-1]:
        nxt = []
        for i in fronts[-1]:
            for j in dominated[i]:
                count[j] -= 1
                if count[j] == 0:
                    nxt.append(j)
        fronts.append(sorted(nxt))
    return [f for f in fronts if f]


def crowding_distance(points, front):
    """{index: crowding distance} within one front; boundary points get inf."""
    dist = {i: 0.0 for i in front}
    if len(front) <= 2:
        return {i: math.inf for i in front}
    for m in range(len(points[front[0]])):
        order = sorted(front, key=lambda i: (points[i][m], i))
        lo, hi = points[order[0]][m], points[order[-1]][m]
        dist[order[0]] = dist[order[-1]] = math.inf
        if hi == lo:
            continue
        for k in range(1, len(order) - 1):
            dist[order[k]] += (points[order[k + 1]][m] - points[order[k - 1]][m]) / (hi - lo)
    return dist


def ranks(points):
    """(front index, crowding distance) per point."""
    out = [None] * len(points)
    for fi, f in enumerate(non_dominated_sort(points)):
        cd = crowding_distance(points, f)
        for i in f:
            out[i] = (fi, cd[i])
    return out


def tournament_key(rank, ident):
    """Sort key for NSGA-II tournaments: front asc, crowding desc, id asc."""
    return rank[0], -rank[1], ident


def nsga2_select(points, ids, mu):
    """Indices of the mu survivors: whole fronts first, the last front cut by crowding
    distance (ties by id), deterministic."""
    keep = []
    for f in non_dominated_sort(points):
        if len(keep) + len(f) <= mu:
            keep += sorted(f, key=lambda i: ids[i])
            continue
        cd = crowding_distance(points, f)
        keep += sorted(f, key=lambda i: (-cd[i], ids[i]))[:mu - len(keep)]
        break
    return keep


def objectives(acc, n_params):
    """(acc, -log2 params): both maximised; a failed genome gets -inf accuracy."""
    a = -math.inf if acc is None else acc
    return a, -math.log2(max(1, n_params or 1))


def front(rows):
    """Non-dominated rows on (acc_agg high, n_params low), fewest params first.
    rows: dicts with genome_id, acc_agg, n_params."""
    pts = [r for r in rows if r.get("acc_agg") is not None and r.get("n_params") is not None]
    out = [r for r in pts if not any(q["acc_agg"] >= r["acc_agg"] and q["n_params"] <= r["n_params"] and
                                     (q["acc_agg"] > r["acc_agg"] or q["n_params"] < r["n_params"]) for q in pts)]
    seen, uniq = set(), []
    for r in sorted(out, key=lambda r: (r["n_params"], -r["acc_agg"], r["genome_id"])):
        if r["genome_id"] not in seen:
            seen.add(r["genome_id"])
            uniq.append({"genome_id": r["genome_id"], "acc_agg": r["acc_agg"], "n_params": r["n_params"]})
    return uniq
