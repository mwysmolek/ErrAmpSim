# ErrAmpSim
Pulsed Er / Er:Yb fibre amplifier and Q-switch designer (Python package, see the module docstrings).

## Frantz–Nodvik pulse amplifier (web app)

`docs/frantz_nodvik.html` is a single-file, dependency-free HTML app that implements the
Frantz–Nodvik gain-saturation model. Open it directly in a browser (no server needed), or
serve the `docs/` folder with GitHub Pages.

- **Input → output**: choose a gain medium (Er / Yb fibre, Nd:YAG, Yb:YAG, Ti:sapphire presets or
  custom cross sections and mode-field diameter), the small-signal gain in dB, a pulse shape
  (Gaussian, super-Gaussian, sech², square, Q-switch-like asymmetric, or a custom table), its FWHM and
  energy or peak power, and see the amplified pulse, the instantaneous gain G(t), the shape at
  successive positions along the amplifier, and an energy sweep.
- **Output → required input**: specify the pulse you want *out* of the amplifier and the app solves
  the closed-form inverse for the seed shape that produces it (the pre-compensation used in shaped-seed
  fibre MOPAs).
- The page ends with a derivation, the numerical method, the model's limits and references.

The app is the textbook plane-wave model with the three-level saturation fluence
hν/(σa+σe); `pulse.fn_deplete` in the Python package generalises the same pass to radial rings,
transverse modes and paired ions, and `pulse.propagate_pulse` adds dispersion, nonlinearity, ASE and
Raman.
