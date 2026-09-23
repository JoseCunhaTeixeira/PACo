"""Pieces shared by the active and passive pipelines."""

from typing import Any

from sigpipe.transformers import Load

from paco.presets import ActivePreset, PassivePreset
from paco.windows import MASWWindow


def load_window(window: MASWWindow) -> Load:
    """Load the window's records, keeping only the window's receivers."""
    return Load(
        file_paths=window.selected_files,
        acquisitions=window.acquisitions,
        data_type="seismic",
        receivers_to_load=window.receiver_indices,
    )


def stage_kwargs(preset: ActivePreset | PassivePreset, stage: str) -> dict[str, Any]:
    """Keyword arguments of the sigpipe transformer running `stage` of `preset`.

    Values derived from the profile must have been filled in by resolve_preset: a None left over
    would otherwise reach sigpipe as a missing argument, deep inside a run.
    """
    kwargs: dict[str, Any] = getattr(preset, stage).model_dump()
    if missing := [name for name, value in kwargs.items() if value is None]:
        raise ValueError(
            f"{stage}.{missing[0]} is not set: resolve the preset against its profile "
            "(paco.presets.resolve_preset) before building pipelines."
        )
    return kwargs
