"""Fitting a preset to a profile: the values PAC's forms derive from the acquisition."""

from typing import Any

from paco.presets.models import ActivePreset, PassivePreset, PresetError
from paco.profiles import Profile

# A switched-on IIR filter without fmax stops just below the Nyquist frequency. PAC's form puts it
# at Nyquist, which sigpipe's filter rejects (it requires fmax < Nyquist).
IIR_FMAX_NYQUIST_FRACTION = 0.95


def resolve_preset[P: ActivePreset | PassivePreset](preset: P, profile: Profile) -> P:
    """`preset` with every value left to None derived from `profile`, and checked against it."""
    if preset.mode != profile.kind:
        raise PresetError(
            f"Preset '{preset.mode}' only fits {preset.mode} profiles, but '{profile.name}' is "
            f"{profile.kind}. Use preset '{profile.kind}'."
        )

    nyquist = profile.nyquist_hz
    values = preset.model_dump()

    if values["muting"]["method"] == "mute" and values["muting"]["tmax"] is None:
        # As PAC's form shows it: the longest record, rounded to 10 ms.
        record_length = round(max(record.duration_s for record in profile.records), 2)
        _derive(values, "muting", "tmin", "tmax", record_length, "s", profile)

    filtering = values["filtering"]
    if filtering["method"] == "iir":
        if filtering["fmax"] is None:
            fmax = IIR_FMAX_NYQUIST_FRACTION * nyquist
            _derive(values, "filtering", "fmin", "fmax", fmax, "Hz", profile)
        elif filtering["fmax"] >= nyquist:
            raise PresetError(
                f"filtering.fmax ({filtering['fmax']:g} Hz) must be below the Nyquist frequency "
                f"of profile '{profile.name}' ({nyquist:g} Hz)."
            )

    whitening = values.get("whitening")  # passive only
    if whitening is not None and whitening["method"] == "onebit_apod" and whitening["fmax"] is None:
        _derive(values, "whitening", "fmin", "fmax", nyquist, "Hz", profile)

    # Validating again runs every other check on the completed values.
    return type(preset).model_validate(values)


def _derive(
    values: dict[str, Any],
    stage: str,
    lower: str,
    upper: str,
    value: float,
    unit: str,
    profile: Profile,
) -> None:
    """Set `stage.upper` to its derived `value`, which must stay above `stage.lower`."""
    params = values[stage]
    if params[lower] >= value:
        raise PresetError(
            f"{stage}.{lower} ({params[lower]:g} {unit}) must be below {stage}.{upper}, which "
            f"defaults to {value:g} {unit} for profile '{profile.name}'. "
            f"Lower {stage}.{lower} or set {stage}.{upper}."
        )
    params[upper] = value
