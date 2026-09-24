"""Which image pipeline each preset builds, as in PAC's adapters/registry.py."""

from collections.abc import Callable
from pathlib import Path

from sigpipe.base import Pipeline

from paco.pipelines.active import build_active_pipeline
from paco.pipelines.passive import build_passive_pipeline
from paco.presets import ActivePreset, PassivePreset
from paco.profiles import ProfileKind
from paco.windows import MASWWindow

PIPELINE_BUILDERS: dict[str, Callable[..., Pipeline]] = {
    ProfileKind.ACTIVE: build_active_pipeline,
    ProfileKind.PASSIVE: build_passive_pipeline,
}


def build_image_pipeline(
    preset: ActivePreset | PassivePreset,
    window: MASWWindow,
    records_folder: Path,
    output_folder: Path,
) -> Pipeline:
    """The pipeline of `preset` for one window, from the preprocessed records in
    `records_folder`, writing its figures and results to `output_folder`."""
    builder = PIPELINE_BUILDERS[preset.mode]
    return builder(
        preset=preset, window=window, records_folder=records_folder, output_folder=output_folder
    )
