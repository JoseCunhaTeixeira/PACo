"""MASW windows: sliding sub-arrays of receivers, each with the records that illuminate it.

Port of PAC's adapters/windows.py on top of paco.profiles: same windows, same shot selection.
"""

from .building import Geometry, apply_exclusions, build_windows
from .models import Exclusions, MASWParameters, MASWWindow

__all__ = [
    "Exclusions",
    "Geometry",
    "MASWParameters",
    "MASWWindow",
    "apply_exclusions",
    "build_windows",
]
