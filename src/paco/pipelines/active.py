"""PAC's active pipeline, from the preprocessed records: a dispersion image per shot, then the
images stacked."""

from pathlib import Path

from sigpipe.base import Pipeline
from sigpipe.transformers import Dispersion, Plot, Save, Stack

from paco.pipelines.common import load_preprocessed, stage_kwargs
from paco.presets import ActivePreset
from paco.windows import MASWWindow


def build_active_pipeline(
    preset: ActivePreset, window: MASWWindow, records_folder: Path, output_folder: Path
) -> Pipeline:
    return (
        load_preprocessed(window, records_folder)
        >> Plot(folder_path=output_folder)
        >> Dispersion(method="phase", **stage_kwargs(preset, "dispersion"))
        >> Stack(method="linear")
        >> Plot(folder_path=output_folder, normalize=True)
        >> Save(folder_path=output_folder)
    )
