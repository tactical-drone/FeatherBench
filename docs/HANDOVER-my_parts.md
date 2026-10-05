<!-- Part of nn-playground. AGPL-3.0; for other licensing see COMMERCIAL.md. -->

# Requested edits to my_parts.py (for the team that owns it)

> Applied in 5a0e5af. Since then Wide has real skip connections (see CHANGELOG,
> "Unreleased"), so the notes below about skip being unused are history.

`my_parts.py` is V's workbench and belongs to your team, so the rsi console team did not
touch it. These are the only edits we ask for. **None of them changes Wide's
architecture, `forward`, its layers, or any number it computes.** Line numbers are from our
(possibly stale) copy, so match by content.

Nothing breaks if you skip them: the rsi console already redirects stray prints to stderr,
and no test depends on these edits. Any edit to `my_parts.py` changes the code fingerprint,
so the rsi cell cache recomputes cells once afterwards. That is expected.

## 1. Delete the debug print (requested)

`Wide.__init__` prints on every model rebuild, that is every knob change in the UI and
every training cell. It floods the UI console and pollutes `python headless.py` stdout,
which scripts parse.

```diff
         self.width = width
         self.hidden_sizes = [n_x, width, n_perc, n_perc, n_perc, n_in, n_perc, n_out]  # one per tensor forward() collects
-        print("Width =", width);
     def describe(self):
```

## 2. A docstring for Wide (recommended)

People and agents keep asking which fields the default model reads, and why `--layers`
does nothing on it. Add this as the first statement of `class Wide`, nothing else changes:

```diff
 class Wide(nn.Module):
+    """The default model ("custom nn"). Dataflow, one Linear per step:
+
+        x -> flank   = expand(x)        EXPANSIONS[expand], scale = fourier_freq
+          -> hid     (width)    + activation
+          -> head    (classes)
+          -> decide  (classes)  + act_decide
+          -> perc    (classes)
+          -> relate  (n_in)     + act_relate
+          -> prepare (classes)  + act_prepare
+          -> focus   (n_out, no bias)
+
+    Reads width, activation, expand, fourier_freq, classes, act_decide, act_relate and
+    act_prepare ("same" = activation). skip is accepted but deliberately unused: forward
+    never calls it. layers and layer are mlp fields; this model ignores them.
+    """
     def __init__(self, n_in, n_out, width, activation, skip="none", expand="linear", classes=5,
```

## 3. Small tidy-ups (optional)

- A comment above the live `add` / `concat` skips, so nobody thinks they are broken (mlp
  now checks them at build time, Wide never calls skip). Keep V's own comments as they are:

  ```diff
   SKIPS.register("half residual", lambda h, x: h + 0.5 * x if h.shape == x.shape else h)
  +# mlp checks at build time that a skip fits: "add" needs equal widths (readable error
  +# otherwise), "concat" widens the next layer. Wide stores skip but never calls it.
   SKIPS.register("add",    lambda h, x: h + x)                   # ResNet style, your l + head(l)
  ```

- `import torch.nn.functional as F` is unused and can go.
- Trailing whitespace in `forward` (the blank first line and the `perc`, `prepare` and
  `out` lines). Leave the `self.decide = ...` line as it is: whitespace only, but the
  architecture check below would flag it.

## Please don't

- Change Wide's layers, `forward`, `hidden_sizes`, `describe()` text or the
  `MODELS.register("custom nn", ...)` line. The default numbers depend on them.
- Delete the commented history lines inside Wide (`# was n_out*2`, the `#d = ...` block).
  They are V's notes.

## How to check

From the repo root:

    python -c "src=open('my_parts.py',encoding='utf-8').read();assert 'print(' not in src.split('class Wide')[1].split('MODELS.register')[0]"
    python -m unittest tests.test_core_equivalence -v
    git diff -- my_parts.py | grep '^[-+]' | grep -E 'nn\.Linear|def forward|self\.(flank|hid|head|perc|relate|prepare|decide|focus)\s*='

The first two must pass; the last must print nothing (architecture untouched).
