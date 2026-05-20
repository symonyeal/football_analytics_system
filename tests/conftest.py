"""Pytest bootstrap.

Some environments restrict access to the system temp directory (the
``pytest-of-<user>`` base under ``%LOCALAPPDATA%\\Temp`` can carry a broken
ACL). To keep ``pytest`` runnable everywhere — including offline CI — point
pytest's temp root at a writable repo-local directory when the default base is
not accessible.
"""

from __future__ import annotations

import getpass
import os
import tempfile
from pathlib import Path

_REPO_TMP = Path(__file__).resolve().parent.parent / ".pytest_tmp"


def _default_base_ok() -> bool:
    base = Path(tempfile.gettempdir())
    try:
        user = getpass.getuser()
    except Exception:
        user = "unknown"
    # pytest creates and scandirs this per-user base; a broken ACL here breaks
    # every tmp_path test even when the parent temp dir is writable.
    pytest_base = base / f"pytest-of-{user}"
    try:
        if pytest_base.exists():
            list(os.scandir(pytest_base))
        probe = base / ".fas_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return os.access(base, os.W_OK)
    except Exception:
        return False


if "PYTEST_DEBUG_TEMPROOT" not in os.environ and not _default_base_ok():
    _REPO_TMP.mkdir(parents=True, exist_ok=True)
    os.environ["PYTEST_DEBUG_TEMPROOT"] = str(_REPO_TMP)
