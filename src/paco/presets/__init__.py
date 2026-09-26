"""Processing presets: PAC's active, passive and passive-active configurations, with PAC's form
defaults.

The preset models are generated from sigpipe's functions and restricted to what PAC uses
(stages.py). The agent picks a preset by name and overrides a few values (make_preset); the
preset is then fitted to a profile, which fills the values PAC derives from the data
(resolve_preset).
"""

from .explaining import explain_parameters
from .making import PRESETS, apply_overrides, make_preset
from .models import (
    ActivePreset,
    PassiveActivePreset,
    PassivePreset,
    Preset,
    PresetBase,
    PresetError,
)
from .resolving import IIR_FMAX_NYQUIST_FRACTION, resolve_preset
from .schemas import override_schema, schema_size, without_titles

__all__ = [
    "IIR_FMAX_NYQUIST_FRACTION",
    "PRESETS",
    "ActivePreset",
    "PassiveActivePreset",
    "PassivePreset",
    "Preset",
    "PresetBase",
    "PresetError",
    "apply_overrides",
    "explain_parameters",
    "make_preset",
    "override_schema",
    "resolve_preset",
    "schema_size",
    "without_titles",
]
