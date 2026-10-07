"""
packaging/build_windows.py // FeatherBench.exe and its release zip.

    pip install pyinstaller
    python packaging/build_windows.py            # -> dist/FeatherBench-v<version>-windows-x64.zip
    python packaging/build_windows.py --smoke    # also start the built exe and train 100 steps

A one-folder PyInstaller build of nn_playground.py (one-file would unpack torch on every
start). Bundled as data: the brand, the rules and leaderboard snapshot, the search space,
and the nncore + my_parts sources, so the exe computes the same code fingerprint as a
source checkout and its FeatherBench numbers compare with everyone else's.

Part of nn-playground. AGPL-3.0; for other licensing see COMMERCIAL.md.
"""
import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAME = "FeatherBench"
REPO_URL = "https://github.com/tactical-drone/FeatherBench"

QUICKSTART = """FeatherBench {version} for Windows (x64)
=========================================

Run FeatherBench.exe. No Python needed.

Recursive Super Intelligence: Featherweight Championship of the world!
Nerds vs AI, place your bets! (RSI = Recursive Super Intelligence.)

Tiny neural nets trained live, plus FeatherBench: fewest params that solves every
pattern wins. Press Space to train, F1 for help. The FeatherBench menu scores your setup,
loads the top-10 setups and explains how to make an attempt.

To submit an attempt or drive everything from scripts and AI agents (the rsi console),
use the source version: {url}

The first start takes a few seconds (it loads PyTorch). Windows SmartScreen may warn
about an unrecognised app because this build isn't code-signed; choose "More info" >
"Run anyway" if you trust the download (check the SHA-256 on the release page).

Source code: {url}/tree/v{version}
License: GNU AGPL v3.0 (LICENSE.txt); commercial licenses: COMMERCIAL.md.
Copyright (C) 2026 tactical-drone.
"""


def version():
    sys.path.insert(0, str(ROOT))
    import nncore
    return nncore.__version__


def make_icon(build):
    """brand/logo-256.png -> build/FeatherBench.ico (Qt writes .ico, no Pillow needed)."""
    from PyQt6 import QtGui
    app = QtGui.QGuiApplication.instance() or QtGui.QGuiApplication(sys.argv[:1])  # noqa: F841
    ico = build / f"{NAME}.ico"
    if not QtGui.QImage(str(ROOT / "brand" / "logo-256.png")).save(str(ico)):
        raise SystemExit("could not write the .ico (Qt's ico image plugin missing?)")
    return ico


def build(ver, smoke):
    work, dist = ROOT / "build", ROOT / "dist"
    work.mkdir(exist_ok=True)
    ico = make_icon(work)
    sep = os.pathsep  # PyInstaller --add-data SRC<pathsep>DEST
    data = [("brand/*.png", "brand"), ("featherbench/README.md", "featherbench"),
            ("featherbench/leaderboard.json", "featherbench"), ("rsi/spaces/*.json", "rsi/spaces"),
            ("nncore/*.py", "nncore"), ("my_parts.py", ".")]
    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--windowed", "--name", NAME,
           "--icon", str(ico), "--distpath", str(dist), "--workpath", str(work / "pyinstaller"),
           "--specpath", str(work), "--collect-submodules", "rsi", "--collect-submodules", "nncore",
           "--collect-submodules", "ui", "--hidden-import", "my_parts"]
    for src, dest in data:
        cmd += ["--add-data", f"{ROOT / src}{sep}{dest}"]
    cmd.append(str(ROOT / "nn_playground.py"))
    print("+", " ".join(cmd[:8]), "...")
    subprocess.run(cmd, check=True, cwd=ROOT)

    app_dir = dist / NAME
    (app_dir / "README.txt").write_text(QUICKSTART.format(version=ver, url=REPO_URL), encoding="utf-8")
    shutil.copy(ROOT / "LICENSE", app_dir / "LICENSE.txt")
    shutil.copy(ROOT / "COMMERCIAL.md", app_dir / "COMMERCIAL.md")

    if smoke:
        exe = app_dir / f"{NAME}.exe"
        env = dict(os.environ, QT_QPA_PLATFORM=os.environ.get("QT_QPA_PLATFORM", "offscreen"))
        code = subprocess.run([str(exe), "--smoke-test", "100"], env=env, timeout=600).returncode
        print(f"smoke test: {exe.name} --smoke-test 100 -> exit {code}")
        if code != 0:
            raise SystemExit(f"smoke test failed (exit {code})")

    zpath = dist / f"{NAME}-v{ver}-windows-x64.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in sorted(app_dir.rglob("*")):
            if f.is_file():
                z.write(f, Path(NAME) / f.relative_to(app_dir))
    print(f"wrote {zpath} ({zpath.stat().st_size / 1e6:.0f} MB)")
    return zpath


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="start the built exe and train 100 steps")
    args = ap.parse_args()
    build(version(), args.smoke)


if __name__ == "__main__":
    main()
