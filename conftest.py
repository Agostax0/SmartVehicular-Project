"""Root conftest: ensure `src/` is on sys.path so bare imports used by the
source code (e.g. ``from utils.kalman import …``) resolve correctly when
pytest is invoked from the project root."""

import sys
from pathlib import Path

_src = str(Path(__file__).resolve().parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)
