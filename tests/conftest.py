"""Every test gets a private data folder and a headless Qt."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("AMPSTUDIO_HOME", tempfile.mkdtemp(prefix="ampstudio-test-"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--no-sandbox --disable-gpu")
os.environ.setdefault("PYTHONPATH", str(ROOT))           # batch workers start with -m fibamp.batch
