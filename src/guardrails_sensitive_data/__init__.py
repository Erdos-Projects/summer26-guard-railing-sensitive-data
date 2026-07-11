"""Netflix Prize privacy experiments."""

from __future__ import annotations

import sys as _sys

from . import data as _data

__all__ = ["__version__"]

__version__ = "0.1.0"

# Backward-compatible module aliases for older notebooks.
_sys.modules.setdefault(__name__ + ".imdb", _data)
_sys.modules.setdefault(__name__ + ".netflix_io", _data)
_sys.modules.setdefault(__name__ + ".synthetic", _data)
