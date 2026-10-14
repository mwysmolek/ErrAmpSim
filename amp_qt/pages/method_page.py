"""Method: what the model does and assumes, straight from the engine's own documentation."""
from __future__ import annotations

import html

from PySide6.QtWidgets import QTextBrowser, QVBoxLayout, QWidget

from fibamp import amplifier, batch, fiber, inversion, modes, pulse, qswitch, sbs, xsections
from fibamp.paths import home_dir

INTRO = """
<h1>kayEi AmpStudio</h1>
<p>Pulsed Er and Er:Yb fibre amplifier and Q-switch designer. One stage at a time: a fibre (datasheet in, doping
and modes out), a pump and a seed pulse. The solver finds the periodic steady state of the pulse train,
propagates the pulse through the inverted fibre with gain saturation, dispersion and nonlinearity, simulates the
backward Brillouin Stokes wave, and checks itself: the photon-number ledger, convergence and window edges are
reported with every result.</p>
<h2>Workflow</h2>
<ol>
<li><b>Fibre</b> — enter the datasheet (or load one from the library). The right side shows the doping, overlaps,
mode areas and gain the datasheet implies: a sanity check before any run.</li>
<li><b>Pump &amp; seed</b> — pump power, wavelength and direction; the seed pulse (analytic shape, a measured CSV,
or the output of a previous stage).</li>
<li><b>▶ Simulate</b> (F5) — the <b>Results</b> page shows the verdict, the output pulse, OSA-style spectra at both
ends, everything along the fibre, SBS and heat. <i>Save stage</i> / <i>Next stage</i> chain amplifiers.</li>
<li><b>Batch</b> — sweep length, pump, seed, fibre… on all cores and rank the cases against an objective and
constraints (e.g. maximise energy with SBS &lt; 1 %).</li>
<li><b>Q-switch</b> — the same fibre and pump in an actively Q-switched cavity; its pulse can seed an amplifier.</li>
</ol>
<h2>Portable</h2>
<p>Everything the app writes (projects, fibre library, batches, stages, settings, log) is in
<code>{home}</code>. Copy the app folder to move it, delete it to remove it: nothing goes into the registry.</p>
<p><b>Cross sections</b>: the shipped tables are representative literature-class spectra, not vendor data. Import
your fibre vendor's spectra (Fibre ▸ Import cross sections) for quantitative work.</p>
<h2>Headless</h2>
<pre>kayei_ampstudio.exe run design.ampstudio --json out.json --stage out.fibamp.npz --csv outdir
kayei_ampstudio.exe batch design.ampstudio
kayei_ampstudio.exe qswitch design.ampstudio --json q.json
kayei_ampstudio.exe new "C-band" design.ampstudio</pre>
<p>The exe is a windowed program: console output goes to <code>ampstudio.log</code> in the data folder, so use the
file outputs in scripts.</p>
"""

SECTIONS = [("The fibre", fiber), ("Transverse modes", modes), ("Cross sections", xsections),
            ("Between pulses: pump, ASE, populations", inversion), ("The pulse", pulse),
            ("Brillouin scattering", sbs), ("The stage", amplifier), ("Q-switched oscillator", qswitch),
            ("Batches", batch)]


def method_html() -> str:
    parts = [INTRO.format(home=html.escape(str(home_dir())))]
    parts.append("<h1>The model</h1>")
    for title, mod in SECTIONS:
        doc = (mod.__doc__ or "").strip()
        if doc:
            parts.append(f"<h2>{html.escape(title)}</h2><pre style='white-space:pre-wrap'>{html.escape(doc)}</pre>")
    return "".join(parts)


class MethodPage(QWidget):
    def __init__(self, win):
        super().__init__(win)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 8, 16, 8)
        self.text = QTextBrowser()
        self.text.setOpenExternalLinks(True)
        self.text.document().setDefaultStyleSheet(
            "h1{font-size:20px;} h2{font-size:15px;color:#22d3ee;margin-top:14px;} "
            "pre{font-family:Consolas,'DejaVu Sans Mono',monospace;font-size:12px;color:#c7d2e0;} "
            "code{font-family:Consolas,monospace;color:#67e8f9;} p,li{line-height:150%;}")
        self.text.setHtml(method_html())
        lay.addWidget(self.text)
