from __future__ import annotations

import os
import site
import sys


def configure_runtime_environment() -> None:
    """Configure deterministic runtime behavior for embeddings stack on Windows.

    - Disable TensorFlow integration in transformers to avoid Keras 3 conflicts.
    - Remove user-site paths from sys.path so Conda/venv packages are not shadowed.
    """
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    user_sites: set[str] = set()
    try:
        raw = site.getusersitepackages()
        if isinstance(raw, str):
            user_sites.add(_normalize_path(raw))
        elif isinstance(raw, (list, tuple, set)):
            user_sites.update(_normalize_path(path) for path in raw)
    except Exception:
        pass

    try:
        site.ENABLE_USER_SITE = False
    except Exception:
        pass

    for path in list(sys.path):
        normalized = _normalize_path(path)
        if normalized in user_sites or _looks_like_windows_user_site(normalized):
            while path in sys.path:
                sys.path.remove(path)


def _normalize_path(path: str) -> str:
    return os.path.normcase(os.path.normpath(path or ""))


def _looks_like_windows_user_site(normalized_path: str) -> bool:
    lowered = normalized_path.lower()
    return (
        "\\appdata\\roaming\\python\\python" in lowered
        and lowered.endswith("\\site-packages")
    )
