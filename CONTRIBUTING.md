# Contributing

<!-- Part of nn-playground. AGPL-3.0; for other licensing see COMMERCIAL.md. -->

Thanks for helping. NN Playground is small on purpose: registries of swappable parts,
a `Config`, a `Session`, a window, and the rsi console for scripts and agents.

## Adding a part

For yourself: register it in `my_parts.py`. It shows up in the UI dropdown, in
`headless.py` and in `python -m rsi describe`, no other file needs editing.

    from nncore import ACTIVATIONS
    ACTIVATIONS.register("snake", Snake, doc="x + sin²(x)")

For a contribution:

- Put it in the matching `nncore/` file (the README compartments table says which file
  and which contract). **Append** it after the existing registrations: registry order is
  the UI dropdown order and the order the console enumerates, so inserting in the middle
  shifts other people's searches.
- Give it a `doc=` line; `describe` shows it to agents.
- A part that needs its own knob reads `cfg.extra["name"]`. Models get `cfg`; schedules
  and train steps get `cfg=` when their signature accepts it (`call_accepting`). Don't add
  a `Config` field without asking first.
- Parts written by AI agents go in `agent_parts/<idea>.py`, loaded with
  `--parts agent_parts.<idea>`, not in `my_parts.py`.
- New files start with a header comment naming the AGPL-3.0 licence and pointing to
  COMMERCIAL.md (copy it from any file in `rsi/`).

Hard rules, the tests enforce them:

- **The default config's numbers must not move.** No change may alter the random draw
  order on the default path. New behaviour is opt-in: a new registry entry, a keyword
  that defaults to the old behaviour.
- Don't change `Config` defaults, `Wide` in `my_parts.py`, LICENSE or COMMERCIAL.md.
- Anything that builds a model outside `Session` (param counts, previews) goes through
  `nncore.schema`, which leaves the torch RNG and stdout alone.

## Running the tests

From the repo root, in the same venv you run the playground with:

    python -m unittest discover -s tests -t . -v      # a few minutes; must be OK before a PR
    python -m unittest tests.test_core_equivalence -v # the numerics guard, if you only have a minute
    python -m rsi doctor                              # environment checks

`tests/test_core_equivalence.py` trains the new `Session` against a frozen copy of the
old one and demands bit-identical weights. If it fails, your change moved the numbers.

The slow golden test (default config, 7000 steps) runs only when asked:

    RSI_SLOW=1 python -m unittest tests.test_golden -v      (PowerShell: $env:RSI_SLOW=1)

Golden numbers depend on the platform and torch version (99.6 / 97.5 on Windows 11,
98.3 / 90.0 on Linux torch 2.14), so it is skipped where `tests/golden.json` has no entry.
`python -m rsi doctor --record-golden` adds one for your machine; don't commit that
unless asked.

## Pull requests

- One idea per PR. Say what changed and how you checked it (a `python -m rsi run ...`
  line with seeds is the best evidence for a new part).
- Add a line to CHANGELOG.md under "Unreleased".
- Tick the licence box in the PR template (below).

## Licence grant

NN Playground is dual-licensed (see COMMERCIAL.md). To keep that possible, every
contribution needs this grant. Ticking the box in the PR template means you agree to it:

<!-- TODO(V): choose ONE of "AGPL-3.0-only" or "AGPL-3.0-or-later" below, say the same in
     README.md and COMMERCIAL.md, then delete this comment. Also consider having a lawyer
     read the paragraph once; it is a plain-language grant, not legal advice. -->

> By submitting a contribution to this project, I agree that it is licensed under
> **AGPL-3.0-[only|or-later]** *(TODO(V): pick one)*, and I grant tactical-drone a
> perpetual, worldwide, non-exclusive, royalty-free, irrevocable licence to use, modify,
> sublicense and distribute it under other licences as well, including commercial ones.
> I confirm the contribution is my own work, or that I have the right to submit it under
> these terms.
