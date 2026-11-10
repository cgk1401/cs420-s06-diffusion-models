"""Locate the Python Typst binding without depending on a shell executable."""
import importlib
import os
from pathlib import Path
import sys
import tempfile


def get_typst():
    try:
        module = importlib.import_module("typst")
    except ImportError:
        module = None
    if module is not None and hasattr(module, "compile"):
        return module
    candidates = [
        os.environ.get("TYPST_PYTHON_PATH"),
        str(Path(tempfile.gettempdir()) / "cs420-report-review-runtime"),
    ]
    for directory in candidates:
        if directory and (Path(directory) / "typst" / "__init__.py").exists():
            sys.path.insert(0, directory)
            sys.modules.pop("typst", None)
            module = importlib.import_module("typst")
            if hasattr(module, "compile"):
                return module
    raise RuntimeError("Install the compiler with: python -m pip install typst")
