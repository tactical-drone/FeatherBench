<!-- Part of nn-playground. AGPL-3.0; for other licensing see COMMERCIAL.md. -->

<!-- FeatherBench attempt? Add only featherbench/submissions/<your GitHub username>.json,
     keep the licence grant below, and skip the rest: CI checks and scores it. -->

## What and why

<!-- One idea per PR. What changed, and why. -->

## How I checked it

<!-- e.g. python -m rsi run --model mlp --activation snake --seeds 0-4 --steps 3000 -> test_acc mean 0.97 -->

## Checklist

- [ ] `python -m unittest discover -s tests -t . -v` is OK
- [ ] Default numbers unchanged (`tests/test_core_equivalence.py` is OK); new behaviour is opt-in
- [ ] New parts are appended after the existing registrations and have a `doc=` line
- [ ] New files carry the licence header; CHANGELOG.md has a line under "Unreleased"

## Licence grant (required)

- [ ] I agree to the licence grant in [CONTRIBUTING.md](../CONTRIBUTING.md#licence-grant): my contribution
      is licensed under the project's AGPL-3.0 licence, and tactical-drone may also distribute it under
      other licences, including commercial ones. I have the right to submit it under these terms.
