# kayEi AmpStudio (ErrAmpSim)

Pulsed **Er / Er:Yb fibre amplifier and Q-switch designer** for Windows, as a **portable** app.
It is the desktop version of `fibamp_app`, built the same way as kayEi RiskStudio and TwinStudio:
a native Qt window, interactive plotly charts and a self-contained PyInstaller build.

**Download:** the latest `kayEi-AmpStudio-<version>-win64-portable.zip` from
[Releases](../../releases). Unzip it anywhere and run `kayei_ampstudio.exe`. Nothing is
installed, and the app writes nothing to the registry or AppData.

## What it does

| Page | |
|---|---|
| 🧵 **Fibre** | Datasheet in (geometry, NA, absorption figures, losses, reflections, modes). Shows what follows from it: doping, overlaps, A_eff, saturation energy, gain ceiling, cross sections, gain spectrum. Includes a fibre library and imports your vendor's cross-section CSV. |
| ⚡ **Pump & seed** | Co- and counter-pump, core or cladding pumping, pulsed pump. Seed is an analytic shape, a measured pulse (CSV) or a previous stage's output. Live preview of the launched pulse and its spectrum. |
| ⚙ **Numerics** | One click for Draft / Standard / Fine, or every grid, convergence and physics switch by hand. |
| 📈 **Results** | Verdict and KPIs, with warnings ranked (SBS runaway, no steady state, parasitic lasing). Charts: pulse, OSA-style spectra at both ends, everything along the fibre, z-evolution maps, transverse inversion and modes, SBS Stokes wave, heat, the inter-pulse trace, power budget. Exports: stage file, CSV, JSON report, *next stage from this output*. |
| 🧮 **Batch** | Sweep length, pump, seed, fibre and more on all cores (each case in its own process). Ranks cases by objective under constraints and builds an HTML report with sensitivities and maps. Any case loads back into the project. |
| 🔦 **Q-switch** | The same fibre and pump in an actively Q-switched cavity. Gives the pulse, a Degnan design map and the hold-off checks; the pulse can seed an amplifier. |
| 📘 **Method** | The model's assumptions, straight from the engine's documentation. |

New projects start from a working preset, because the engine's bare defaults don't form a usable design:

- L-band core-pumped stage (Er80-8/125 class, the fibre the L-band shortlist study picked)
- C-band pre-amplifier (Er80-4/125 class)
- Er:Yb 25/300 cladding-pumped booster

## Portable data

Everything the app writes goes to `data\` next to the exe:

- projects (`.ampstudio`)
- the fibre library
- batches and their reports
- stage files
- settings
- `ampstudio.log`

To move the app, copy the folder. To update, unzip the new version and copy `data\` across.
To uninstall, delete the folder. If the folder isn't writable (for example under Program
Files), the app falls back to `%LOCALAPPDATA%\kayEi AmpStudio`.

## Headless

```bat
kayei_ampstudio.exe new "C-band" design.ampstudio
kayei_ampstudio.exe run design.ampstudio --accuracy draft --json out.json --stage out.fibamp.npz --csv outdir
kayei_ampstudio.exe batch design.ampstudio --axis length_m=2:6:1 --axis pump_co_W=0.5,1
kayei_ampstudio.exe qswitch design.ampstudio --json q.json
```

The exe is windowed, so console output goes to `data\ampstudio.log`; use the file outputs in
scripts. From source: `python -m fibamp.cli …`.

## Repository

```
fibamp/        physics engine (the fibamp_app package, unchanged physics) + paths, presets, project, cli
amp_qt/        desktop app (PySide6 + Qt WebEngine for plotly), kayEi theme shared with the studios
packaging/     PyInstaller build -> dist/kayEi-AmpStudio/ + portable zip (Windows)
tests/         engine, batch workers, portability, the desktop app (headless)
.github/       Windows CI: tests, build, smoke test of the exe, headless run + batch, portability
               check, zip, release
```

The engine's physics files are the originals from `fibamp_app` (git history keeps them as
renames). The only engine changes are about where files go and how batch workers start
inside a frozen exe:

- `io.py` and `batch.py` keep files in the portable data folder instead of `~/fibamp`.
- `batch.py` passes jobs and results to workers through files (a windowed exe has no stdout), re-launches `kayei_ampstudio.exe --batch-worker` when frozen, opens no console windows, and can be stopped (finished cases are kept and resumed).

Run from source (Windows: `run_ampstudio.bat`):

```
pip install -r requirements.txt
python -m amp_qt.app
python -m pytest -q tests
pip install -r packaging/requirements-build.txt && python packaging/build_exe.py
```

Releases: bump `amp_qt/__init__.py::__version__`. A push to `main` with a new version (or a
`v*` tag) builds, tests and publishes the portable zip.

Cross sections shipped in `fibamp/data` are representative literature-class spectra, **not
vendor data**. For quantitative work, import your fibre vendor's spectra.
