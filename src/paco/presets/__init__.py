"""Processing presets: PAC's active and passive configurations, with PAC's form defaults.

The preset models are generated from sigpipe's functions and restricted to what PAC uses
(stages.py). The agent picks a preset by name and overrides a few values (make_preset); the
preset is then fitted to a profile, which fills the values PAC derives from the data
(resolve_preset).
"""

from .making import PRESETS, make_preset
from .models import ActivePreset, PassivePreset, Preset, PresetBase, PresetError
from .resolving import IIR_FMAX_NYQUIST_FRACTION, resolve_preset
from .schemas import override_schema, schema_size, without_titles

__all__ = [
    "IIR_FMAX_NYQUIST_FRACTION",
    "PRESETS",
    "ActivePreset",
    "PassivePreset",
    "Preset",
    "PresetBase",
    "PresetError",
    "make_preset",
    "override_schema",
    "resolve_preset",
    "schema_size",
    "without_titles",
]
