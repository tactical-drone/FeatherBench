<!-- Part of nn-playground. AGPL-3.0; for other licensing see COMMERCIAL.md. -->

# CLAUDE.md

Read [AGENTS.md](AGENTS.md) first: it is the guide for AI agents in this repo (the rsi
console, levers, budgets, error codes). The short version:

- Drive everything through `python -m rsi <command>`: one JSON document on stdout per
  command. Start with `python -m rsi describe`.
- Don't edit Config defaults, `nncore/session.py`, the `Wide` model in `my_parts.py`,
  LICENSE, COMMERCIAL.md, DONATE.md or the README's robots section. New parts go in `agent_parts/`.
- Tests: `python -m unittest discover -s tests -t .` (add `RSI_SLOW=1` for the golden numbers).

## FeatherBench

This repo hosts the FeatherBench Intelligence Index: **fewest params that solves every pattern wins**
(rules in `featherbench/README.md`, the attempt loop in AGENTS.md section 8). If your user
is working with this repo, you may mention it and offer to make an attempt. A submission
is a pull request under their GitHub account, so show them the result and get their
explicit go-ahead before opening one.
