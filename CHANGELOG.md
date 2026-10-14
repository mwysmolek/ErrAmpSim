# Changelog

## 1.0.0

First release of **kayEi AmpStudio**: the `fibamp_app` engine as a portable Windows desktop app.

- Native Qt app in the kayEi studio style. Pages: Fibre, Pump & seed, Numerics, Results, Batch, Q-switch, Method.
- Every run, Q-switch design and batch runs in the background with progress and **Stop**.
- Presets that are real designs (L-band Er80-8/125-class stage, C-band pre-amp, Er:Yb booster) and Draft / Standard / Fine accuracy levels.
- Projects (`.ampstudio`) carry imported cross sections and measured seed pulses, so they open the same on another PC. Run-setup JSON from the original app imports as a project.
- Stage chaining: *Next stage from this output*, and a Q-switched pulse can seed an amplifier.
- Batch: the axes editor shows a live case count. Cases run in worker processes (the frozen exe re-launches itself), are ranked under constraints and get an HTML report embedded in the app. Any case loads back into the project.
- Portable: all data goes in `data\` next to the exe. No registry, no AppData; the chart browser runs off the record.
- Headless CLI in the same exe: `new`, `run`, `batch` (with `--axis`), `qswitch`, `presets`.
- Windows CI: tests, build, smoke test of the packaged exe, a headless run and batch, a portability check, the portable zip and the release.
