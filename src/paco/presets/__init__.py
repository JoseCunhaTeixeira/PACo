"""Processing presets: PAC's active and passive configurations, with PAC's form defaults.

The agent picks a preset by name and overrides a few values (make_preset); the preset is then
fitted to a profile, which fills the values PAC derives from the data (resolve_preset). The
stage models live in paco.presets.models.
"""

from .making import PRESETS, make_preset
from .models import ActivePreset, PassivePreset, Preset, PresetError
from .resolving import IIR_FMAX_NYQUIST_FRACTION, resolve_preset

__all__ = [
    "IIR_FMAX_NYQUIST_FRACTION",
    "PRESETS",
    "ActivePreset",
    "PassivePreset",
    "Preset",
    "PresetError",
    "make_preset",
    "resolve_preset",
]
