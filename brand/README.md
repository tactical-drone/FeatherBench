<!-- Part of nn-playground. AGPL-3.0; for other licensing see COMMERCIAL.md. -->

# FeatherBench brand

Three neon feathers on deep indigo: pink, lavender, mint. The colours were sampled from
the original feather picture (each feather's lit pixels, its brightest 15 % and the
background); the rendered logo matches it within 5/255 per channel.

| Role | Hex |
|---|---|
| Pink feather (body / highlight / glow) | `#f097c2` / `#f8a1cd` / `#ff3295` |
| Lavender feather | `#c2c3ff` / `#cbcdff` / `#7274ff` |
| Mint feather (also "Bench" in the wordmark) | `#a4f7ed` / `#b2fff7` / `#43ffe8` |
| Background (center / edge) | `#29283c` / `#1b1a29` |
| Wordmark / tagline | `#ecebff` / `#9a97c0` |

| File | Use |
|---|---|
| `logo.svg`, `logo-256.png`, `icon-64.png` | app icon, avatars, favicons |
| `feathers.svg` | the mark alone, transparent background |
| `wordmark.svg`, `wordmark.png` | README header, banners |
| `social-preview.svg`, `social-preview.png` | GitHub social preview (1280 x 640): Settings > General > Social preview |

`python brand/build.py` regenerates everything from one palette (PNG text uses the system
UI font; run it on a desktop session, not headless).
