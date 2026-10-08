<!-- Part of nn-playground. AGPL-3.0; for other licensing see COMMERCIAL.md. -->

# What FeatherBench takes from RSIGym

*RSIGym: A Flexible Environment for Recursive Self-Improvement* (Meng, Du et al., Evolvent
AI / NUS, arXiv 2610.10310, Oct 2026) is an environment where research agents improve a
target model through services for training, inference, evaluation and sandboxes, under a
budget and permissions. FeatherBench is a much smaller world (tiny nets on 2-D patterns,
CPU only), but the same research loop, so several of its ideas carry over directly.

| RSIGym | FeatherBench |
|---|---|
| **RSI = accepted changes carried into later cycles**; recursive progress = inherited changes improve how later improvements are made | `rsi featherbench start` begins from the current leader's setup; `stamp` records the `parent`, and the leaderboard shows each entry's `gain_over_parent` |
| **RSI-Index**: mean share of the remaining gap closed, (s − s₀) / (1 − s₀), per benchmark | `rsi_index`: the same formula per pattern (fresh accuracy vs the default setup, frozen per benchmark version in `featherbench/baselines/`), averaged over the 12 patterns; next to `feather_score`, which ranks by params |
| **Recorded execution**: the services keep their own ledger, independent of what the agent reports | `compute` in the attempt: trials, cells and CPU seconds the rsi console recorded since `start`, alongside the self-reported model, tokens and cost |
| **Budget as part of the problem**, visible to the agent | Steps-based budgets everywhere, a time-box in the agentic prompt (45 min / 40 runs), cached cells so repeats are free |
| **Inspectable evidence**: per-task rewards and trajectories, not one number | `per_dataset` accuracies, `start`'s weakest patterns, decision-map PNGs (`--png`, `--answer-png`) |
| **Everything as a Service**: agents spend their time on research, not infrastructure | The rsi console: one JSON document per command, every lever described by `describe` |
| **Separate final evaluation** under a fixed protocol | CI re-scores every merged submission on Linux, cache off; only that run reaches the leaderboard |
| **Budget hacking** observed (an agent bypassed the budget via found credentials) | Submissions are data only, CI has no secrets and a read-only token on pull requests, and scores are re-run, so a local shortcut can't inflate a score |

What the study found about *how* agents improve systems, which the AGENTS.md loop now
follows:

- The best agents diagnosed failures from evidence before changing anything, and kept
  refining; the agents that shipped quickly scored lower.
- **Heavier is not better**: heavier training updates scored lower in 8 of 10 matched
  comparisons. FeatherBench shows the same dark art (more width or classes can make a
  pattern worse), so the loop says: one light change at a time, compared on the same seeds.
- Doubling the budget gave mixed results; spend did not explain the ranking.

One gap RSIGym states openly (development and final scores use tasks the agent has seen)
FeatherBench now closes: CI also scores every merged submission on secret held-out seeds,
publishes only the aggregates, and flags setups that do clearly worse there (`overfit`).
