"""PAC's active pipeline: a dispersion image per shot, then the images stacked."""

from pathlib import Path

from paco.pipelines.common import load_window, stage_kwargs
from paco.presets import ActivePreset
from paco.windows import MASWWindow
from sigpipe.base import Pipeline
from sigpipe.transformers import Detrend, Dispersion, Filter, Mute, Pad, Plot, Save, Stack


def build_active_pipeline(
    preset: ActivePreset, window: MASWWindow, output_folder: Path
) -> Pipeline:
    return (
        load_window(window)
        >> Detrend(method="constant")
        >> Detrend(method="linear")
        >> Mute(**stage_kwargs(preset, "muting"))
        >> Filter(**stage_kwargs(preset, "filtering"))
        >> Plot(folder_path=output_folder)
        >> Pad(n=1_000, taper=25)
        >> Dispersion(method="phase", **stage_kwargs(preset, "dispersion"))
        >> Stack(method="linear")
        >> Plot(folder_path=output_folder, normalize=True)
        >> Save(folder_path=output_folder)
    )
