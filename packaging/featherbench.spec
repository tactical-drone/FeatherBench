# PyInstaller spec for the FeatherBench release. Part of nn-playground (AGPL-3.0; see COMMERCIAL.md).
#
#   pyinstaller packaging/featherbench.spec --noconfirm
#
# Builds one folder, dist/FeatherBench/, with two programs sharing one copy of torch and Qt:
#   FeatherBench.exe      the window (nn_playground.py)
#   featherbench-cli.exe  the rsi console (headless.py <command> == python -m rsi <command>)
import os
from PyInstaller.utils.hooks import collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
ICON = os.path.join(ROOT, "brand", "logo-256.png")  # Pillow converts it to .ico on Windows

hidden = (collect_submodules("nncore") + collect_submodules("rsi") + collect_submodules("ui")
          + ["my_parts"])
datas = [
    (os.path.join(ROOT, "brand", "*.png"), "brand"),
    (os.path.join(ROOT, "rsi", "spaces"), os.path.join("rsi", "spaces")),
    (os.path.join(ROOT, "featherbench", "leaderboard.json"), "featherbench"),
    (os.path.join(ROOT, "tests", "golden.json"), "tests"),
    (os.path.join(ROOT, "LICENSE"), "."),
    (os.path.join(ROOT, "COMMERCIAL.md"), "."),
    (os.path.join(ROOT, "README.md"), "."),
]
excludes = ["tkinter", "matplotlib", "IPython", "pytest"]


def analysis(script):
    return Analysis([os.path.join(ROOT, script)], pathex=[ROOT], hiddenimports=hidden, datas=datas,
                    excludes=excludes, noarchive=False)


gui = analysis("nn_playground.py")
cli = analysis("headless.py")

gui_exe = EXE(PYZ(gui.pure), gui.scripts, [], exclude_binaries=True, name="FeatherBench",
              console=False, icon=ICON)
cli_exe = EXE(PYZ(cli.pure), cli.scripts, [], exclude_binaries=True, name="featherbench-cli",
              console=True, icon=ICON)

COLLECT(gui_exe, gui.binaries, gui.datas, cli_exe, cli.binaries, cli.datas, name="FeatherBench")
