"""
Compartment: SAMPLERS  (which training points each step sees)

Contract:  factory(n, batch_size) -> next_batch()
  next_batch() -> LongTensor of row indices, or None for "use every row".
  batch_size is None for full batch; otherwise already clamped to 1..n.
The factory is re-created whenever data, model or batch settings change, so it
may keep state (e.g. an epoch permutation).
"""
import torch

from .registry import Registry

SAMPLERS = Registry("sampler", "factory(n, batch_size) -> next_batch() -> idx | None")


@SAMPLERS.register("random (with replacement)")
def with_replacement(n, bs):
    if bs is None or bs >= n:
        return lambda: None
    return lambda: torch.randint(0, n, (bs,))


@SAMPLERS.register("epoch shuffle")
def epoch_shuffle(n, bs):
    if bs is None or bs >= n:
        return lambda: None
    state = {"perm": torch.randperm(n), "pos": 0}

    def next_batch():
        if state["pos"] + bs > n:
            state["perm"], state["pos"] = torch.randperm(n), 0
        i = state["pos"]
        state["pos"] += bs
        return state["perm"][i:i + bs]
    return next_batch
