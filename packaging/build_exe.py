"""Build kayEi AmpStudio for Windows as a portable app (PyInstaller, no installer).

    pip install -r requirements.txt -r packaging/requirements-build.txt
    python packaging/build_exe.py            # -> dist/kayEi-AmpStudio/ (the app folder)
                                             #  + dist/kayEi-AmpStudio-<ver>-win64-portable.zip
    python packaging/build_exe.py --no-zip   # the folder only
    python packaging/build_exe.py --zip-only # zip an existing dist/kayEi-AmpStudio/ folder
    python packaging/build_exe.py --console  # debug build with a console window

Portable means: unzip anywhere (Desktop, a network share, a USB stick) and run
kayei_ampstudio.exe. Projects, the fibre library, batches, settings and the log go to the
``data`` folder next to the exe; nothing is written to the registry or AppData. If that
folder is not writable (e.g. unpacked under Program Files) the app falls back to
%LOCALAPPDATA%\\kayEi AmpStudio.

PyInstaller does not cross-compile: run this on Windows (the GitHub workflow
.github/workflows/build-windows.yml does). On another OS it builds that OS's folder, which
is only useful for testing the packaging.
"""
from __future__ import annotations

import os
import re
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "packaging"
BUILD = ROOT / "build"
DIST = ROOT / "dist"
SEP = os.pathsep                 # PyInstaller --add-data separator: ";" on Windows, ":" elsewhere
NAME = "kayei_ampstudio"         # the exe
FOLDER = "kayEi-AmpStudio"       # the app folder (and the folder inside the zip)
TITLE = "kayEi AmpStudio"


def version() -> str:
    text = (ROOT / "amp_qt" / "__init__.py").read_text(encoding="utf-8")
    return re.search(r'__version__\s*=\s*"([^"]+)"', text).group(1)


def make_icon() -> Path:
    from PIL import Image
    BUILD.mkdir(exist_ok=True)
    ico = BUILD / "ampstudio.ico"
    Image.open(ROOT / "assets" / "logo.png").convert("RGBA").save(
        ico, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    return ico


def version_file(ver: str) -> Path:
    """The Windows 'Details' tab of the exe: product name, version, publisher."""
    nums = tuple((list(map(int, re.findall(r"\d+", ver))) + [0, 0, 0, 0])[:4])
    text = f"""VSVersionInfo(
  ffi=FixedFileInfo(filevers={nums}, prodvers={nums}, mask=0x3f, flags=0x0, OS=0x40004,
                    fileType=0x1, subtype=0x0, date=(0, 0)),
  kids=[StringFileInfo([StringTable('040904B0', [
          StringStruct('CompanyName', 'kayEi'),
          StringStruct('FileDescription', '{TITLE} (portable)'),
          StringStruct('FileVersion', '{ver}'),
          StringStruct('InternalName', '{NAME}'),
          StringStruct('OriginalFilename', '{NAME}.exe'),
          StringStruct('ProductName', '{TITLE}'),
          StringStruct('ProductVersion', '{ver}')])]),
        VarFileInfo([VarStruct('Translation', [1033, 1200])])])
"""
    BUILD.mkdir(exist_ok=True)
    path = BUILD / "version_info.txt"
    path.write_text(text, encoding="utf-8")
    return path


# Qt modules the app does not use (QtWebEngine needs QtWebChannel, QtPositioning, QtQml/QtQuick: keep those)
QT_EXCLUDES = ("PySide6.Qt3DCore", "PySide6.Qt3DRender", "PySide6.QtMultimedia", "PySide6.QtCharts",
               "PySide6.QtDataVisualization", "PySide6.QtBluetooth", "PySide6.QtSensors", "PySide6.QtSerialPort",
               "PySide6.QtTest", "PySide6.QtDesigner", "PySide6.QtHelp", "PySide6.QtNfc", "PySide6.QtRemoteObjects",
               "PySide6.QtScxml", "PySide6.QtSql", "PySide6.QtStateMachine", "PySide6.QtTextToSpeech",
               "PySide6.QtUiTools", "PySide6.QtGraphs", "PySide6.QtHttpServer", "PySide6.QtSpatialAudio",
               "PySide6.QtPdf", "PySide6.QtPdfWidgets", "PySide6.QtQuick3D", "PySide6.QtLocation")
PY_EXCLUDES = ("pytest", "IPython", "matplotlib", "pandas", "PyInstaller", "notebook", "jupyter_client", "tkinter",
               "PIL", "streamlit", "altair", "tornado", "sympy", "numba")


# Qt libraries nothing in the app links against (Qt WebEngine needs Core, Gui, Widgets, Network, OpenGL,
# Positioning, PrintSupport, Qml, QmlMeta, QmlModels, QmlWorkerScript, Quick, QuickWidgets, WebChannel, DBus).
# PySide6's hook copies all of Qt; dropping these takes ~2/3 off the download.
PRUNE_QT = ("3D", "Quick3D", "Charts", "DataVisualization", "Graphs", "Multimedia", "Pdf", "Location",
            "QuickControls2", "QuickDialogs2", "QuickTemplates2", "QuickLayouts", "QuickParticles", "QuickShapes",
            "QuickTimeline", "QuickVectorImage", "QuickEffects", "Labs", "Designer", "Sensors", "SerialPort",
            "SerialBus", "Bluetooth", "Nfc", "RemoteObjects", "Scxml", "Sql", "StateMachine", "TextToSpeech",
            "VirtualKeyboard", "SpatialAudio", "ShaderTools", "Test", "Help", "UiTools", "Lottie", "Protobuf", "Grpc",
            "HttpServer", "WebView", "NetworkAuth", "QmlLocalStorage", "QmlXmlListModel", "QmlNetwork",
            "QmlDesignSupport", "QmlCompiler", "QuickTest", "Charts", "Graphs")
PRUNE_PLUGINS = ("multimedia", "sqldrivers", "sensors", "position", "geometryloaders", "sceneparsers", "renderers",
                 "renderplugins", "qmltooling", "canbus", "texttospeech", "assetimporters", "designer",
                 "virtualkeyboard", "qmllint", "help", "webview", "scxmldatamodel")


def _qt_module(name: str) -> str | None:
    """'Qt6Charts.dll' / 'libQt6Charts.so.6' / 'QtCharts.pyd' / 'QtCharts.abi3.so' -> 'Charts'."""
    m = re.match(r"^(?:lib)?Qt6?([A-Za-z0-9]+?)(?:d)?(?:\.abi3)?\.(?:dll|pyd|so|pyi|dylib)", name)
    return m.group(1) if m else None


def prune(folder: Path) -> None:
    """Remove the parts of Qt the app never loads (see PRUNE_QT / PRUNE_PLUGINS)."""
    before = sum(f.stat().st_size for f in folder.rglob("*") if f.is_file())
    for f in list(folder.rglob("*")):
        if not f.exists():
            continue
        if f.is_dir():
            if f.name == "qml" and f.parent.name in ("PySide6", "Qt"):
                shutil.rmtree(f)                    # QML imports: the app has no QML
            elif f.parent.name == "plugins" and f.name in PRUNE_PLUGINS:
                shutil.rmtree(f)
            elif f.name == "translations" and f.parent.name in ("PySide6", "Qt"):
                for t in f.iterdir():                # keep Chromium's locales, drop Qt's UI translations
                    if t.is_file() and t.suffix == ".qm":
                        t.unlink()
            continue
        mod = _qt_module(f.name)
        if mod and any(mod == p or mod.startswith(p) for p in PRUNE_QT):
            f.unlink()
    after = sum(f.stat().st_size for f in folder.rglob("*") if f.is_file())
    print(f"pruned Qt: {before / 1e6:.0f} MB -> {after / 1e6:.0f} MB")


def build_app(console: bool = False) -> Path:
    import PyInstaller.__main__
    ver = version()
    out = DIST / FOLDER
    args = [
        str(PKG / "ampstudio_desktop.py"),
        "--name", NAME,
        "--onedir", "--contents-directory", "_internal",
        "--noconfirm", "--clean",
        "--icon", str(make_icon()),
        *(["--version-file", str(version_file(ver))] if sys.platform == "win32" else []),
        "--paths", str(ROOT),
        "--add-data", f"{ROOT / 'assets'}{SEP}assets",
        "--add-data", f"{ROOT / 'fibamp' / 'data'}{SEP}fibamp/data",
        "--hidden-import", "PySide6.QtWebEngineWidgets", "--hidden-import", "PySide6.QtWebEngineCore",
        "--hidden-import", "PySide6.QtNetwork",
        "--collect-data", "plotly",
        "--collect-submodules", "plotly",
        "--copy-metadata", "numpy", "--copy-metadata", "plotly", "--copy-metadata", "scipy",
        *[x for m in (*PY_EXCLUDES, *QT_EXCLUDES) for x in ("--exclude-module", m)],
        "--distpath", str(BUILD / "pyi-dist"),
        "--workpath", str(BUILD / "pyi-work"),
        "--specpath", str(BUILD),
        "--console" if console else "--windowed",
    ]
    PyInstaller.__main__.run(args)
    built = BUILD / "pyi-dist" / NAME
    shutil.rmtree(out, ignore_errors=True)
    DIST.mkdir(exist_ok=True)
    shutil.move(str(built), str(out))
    prune(out)
    (out / "README.txt").write_text(readme(ver), encoding="utf-8")
    print(f"app folder: {out}")
    return out


def readme(ver: str) -> str:
    return f"""{TITLE} {ver} - portable

Pulsed Er / Er:Yb fibre amplifier and Q-switch designer.

Run:        kayei_ampstudio.exe
Your data:  the "data" folder next to the exe (created on first start): projects,
            fibre library, batches, stage files, settings and the log.
Move it:    copy this whole folder. Remove it: delete this folder. Nothing is installed,
            nothing goes into the registry.
Update:     unzip the new version and copy your old "data" folder into it.

Headless (scripts / CI; output goes to data\\ampstudio.log):
  kayei_ampstudio.exe new "C-band" design.ampstudio
  kayei_ampstudio.exe run design.ampstudio --json out.json --stage out.fibamp.npz --csv outdir
  kayei_ampstudio.exe batch design.ampstudio --axis length_m=2:6:1 --axis pump_co_W=0.5,1
  kayei_ampstudio.exe qswitch design.ampstudio --json q.json

Windows SmartScreen may warn on first start because the exe is not code-signed:
"More info" -> "Run anyway".
"""


def make_zip(folder: Path) -> Path:
    ver = version()
    plat = "win64" if sys.platform == "win32" else sys.platform
    out = DIST / f"kayEi-AmpStudio-{ver}-{plat}-portable.zip"
    out.unlink(missing_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in sorted(folder.rglob("*")):
            rel = f.relative_to(folder)
            if rel.parts and rel.parts[0] == "data":        # never ship a developer's data folder
                continue
            z.write(f, Path(FOLDER) / rel)
    print(f"portable zip: {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return out


def main() -> None:
    argv = sys.argv[1:]
    if sys.platform != "win32":
        print("note: AmpStudio ships for Windows; this builds a test folder for this OS", flush=True)
    if "--zip-only" in argv:                      # CI: zip the folder it has just tested
        make_zip(DIST / FOLDER)
        return
    folder = build_app(console="--console" in argv)
    if "--no-zip" not in argv:
        make_zip(folder)


if __name__ == "__main__":
    main()
