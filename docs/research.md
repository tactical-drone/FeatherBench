<!-- Part of nn-playground. AGPL-3.0; for other licensing see COMMERCIAL.md. -->

# Research assimilated into FeatherBench

What we took from each paper, and what it measured here. Numbers are local runs of the
official benchmark (`featherbench-general-v1`, 12 patterns x 3 seeds, Windows); CI's Linux
numbers on the leaderboard are the official ones.

## RSIGym (arXiv 2610.10310)

An environment where research agents improve a model under budgets and permissions. We took
inheritance (`featherbench start`), the RSI-Index (`rsi_index`), a recorded compute ledger,
held-out evaluation (secret seeds) and its lessons on how agents improve systems. Details:
[rsigym.md](rsigym.md).

## Recurrent Self-Improvement: cross-loop on-policy distillation (arXiv 2610.10623)

Looped language models reuse one shared block several times: more computation, same
parameters. The paper's D-LoopOPD trains an intermediate loop toward the model's own last
loop (reverse KL, stop-gradient, teacher refreshed every step), so the model is its own
teacher.

What we built:

- `model: "looped"`: input expansion, an input layer, one shared block applied
  `extra.loops` times (input re-injected each loop), one shared readout. Parameters don't
  depend on the number of loops.
- `train_step: "cross-loop distill"`: task loss on the last loop + `extra.distill_weight`
  (default 1) x KL(loop `extra.distill_loop`, default the middle one || last loop, detached).

| Setup | Solved | params_max | Mean acc | Worst | feather_score |
|---|---|---|---|---|---|
| Default (custom nn, the house entry) | 9/12 | 346 | 91.2% | 77.6% | 762.4 |
| looped, width 8, 4 loops | **12/12** | **310** | 94.9% | 91.6% | **984.0** |
| looped w8 L4 + cross-loop distill | 12/12 | 310 | 94.8% | 91.4% | 984.0 |
| looped w8 L4, tanh | 11/12 | 310 | 94.1% | 85.2% | 918.5 |
| looped w8 L4, tanh + cross-loop distill | 12/12 | 310 | 94.7% | 91.0% | 984.0 |
| looped w12 L6, tanh + cross-loop distill | 12/12 | 486 | 94.6% | 91.9% | 982.8 |
| looped w12 L6, sin + cross-loop distill | 7/12 | 486 | 81.2% | 49.5% | 600.9 |

Weight sharing is the big win: the looped net solves every pattern with fewer parameters
than the default. Self-distillation is neutral with softplus and helps with tanh (11 to 12
solved, worst pattern 85 to 91%), consistent with the paper's claim that the deeper loop is
a useful teacher, though not a universal gain at this scale.

## A Fast Algorithm for Maltsev Constraints (arXiv 2610.08207)

Complexity theory: constraint problems that generalise linear equations (Maltsev CSPs)
solved in O(n^2 m) instead of O(n^4 m). No neural technique to borrow, but one connection:
FeatherBench's "Random lines (parity)" pattern is linear equations mod 2, the Maltsev case.
We added the complexity family `parity[k]` (k random lines), so
`python -m rsi complexity --family parity --sizes 1-6` measures the parameters a net needs as
the system grows, on the one family whose exact algorithmic complexity is known: an
empirical exponent next to a real big-O.
