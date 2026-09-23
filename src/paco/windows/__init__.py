"""MASW windows: sliding sub-arrays of receivers, each with the records that illuminate it.

Port of PAC's adapters/windows.py on top of paco.profiles: same windows, same shot selection.
"""

from .building import build_windows
from .models import MASWParameters, MASWWindow

__all__ = ["MASWParameters", "MASWWindow", "build_windows"]
