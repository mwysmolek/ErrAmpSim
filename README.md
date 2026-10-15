# ErrAmpSim
Pulsed Er / Er:Yb fibre amplifier and Q-switch designer (Python package, see the module docstrings).

## Frantz–Nodvik pulse amplifier (web app)

`docs/frantz_nodvik.html` is a single-file, dependency-free HTML app that implements the
Frantz–Nodvik gain-saturation model. Open it directly in a browser (no server needed), or
serve the `docs/` folder with GitHub Pages.

- **Fibre datasheet mode**: enter what the datasheet gives (core or cladding absorption at a
  reference wavelength, core and cladding diameter, NA or MFD, length, signal wavelength) and the app
  derives the dopant concentration, LP01 mode size and overlap, saturation energy and small-signal gain
  from the shipped cross-section tables, the same way `fiber.py` does. The gain comes from an average
  inversion you set, or from a lumped pump estimate (pump power, wavelength, core or cladding launch,
  repetition rate, lifetime, ASE gain cap) solved for the periodic steady state.
- **E_sat & gain mode**: choose a bulk or fibre preset (Er / Yb fibre, Nd:YAG, Yb:YAG, Ti:sapphire) or
  custom cross sections and beam diameter, and enter the small-signal gain in dB directly.
- **Input → output**: pick a pulse shape (Gaussian, super-Gaussian, sech², square, Q-switch-like
  asymmetric, or a custom table), its FWHM and energy or peak power, and see the amplified pulse, the
  pulse width before and after, the instantaneous gain G(t), the shape at successive positions along
  the amplifier, and an energy sweep.
- **Output → required input**: specify the pulse you want *out* of the amplifier and the app solves
  the closed-form inverse for the seed shape that produces it (the pre-compensation used in shaped-seed
  fibre MOPAs).
- The page ends with a derivation, the numerical method, the model's limits and references.

The app is the textbook plane-wave model with the three-level saturation fluence
hν/(σa+σe); `pulse.fn_deplete` in the Python package generalises the same pass to radial rings,
transverse modes and paired ions, and `pulse.propagate_pulse` adds dispersion, nonlinearity, ASE and
Raman.
