"""Picking a preset by name and applying the agent's overrides to it."""

from collections.abc import Mapping

from paco.presets.models import ActivePreset, PassivePreset, PresetError
from paco.profiles import ProfileKind

PRESETS: dict[str, type[ActivePreset] | type[PassivePreset]] = {
    ProfileKind.ACTIVE: ActivePreset,
    ProfileKind.PASSIVE: PassivePreset,
}


def make_preset(
    name: str, overrides: Mapping[str, object] | None = None
) -> ActivePreset | PassivePreset:
    """The named preset with `overrides` applied.

    Overrides are a partial preset, e.g. {"filtering": {"method": "iir", "fmin": 5}}: stages
    and fields left out keep PAC's defaults. Invalid overrides raise pydantic's ValidationError.
    """
    preset = PRESETS.get(name)
    if preset is None:
        raise PresetError(f"Unknown preset '{name}'. Available presets: {', '.join(PRESETS)}.")
    return preset.model_validate(dict(overrides or {}))
