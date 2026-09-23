"""PAC's passive pipeline: noise segments cross-correlated and stacked, then a dispersion image."""

from pathlib import Path

from sigpipe.base import Pipeline
from sigpipe.transformers import (
    Apodize,
    Correlate,
    Detrend,
    Dispersion,
    Filter,
    Mute,
    Normalize,
    Pad,
    Plot,
    Save,
    Selection,
    Slice,
    Stack,
    Whiten,
)

from paco.pipelines.common import load_window, stage_kwargs
from paco.presets import PassivePreset
from paco.windows import MASWWindow


def build_passive_pipeline(
    preset: PassivePreset, window: MASWWindow, output_folder: Path
) -> Pipeline:
    return (
        load_window(window)
        >> Detrend(method="constant")
        >> Detrend(method="linear")
        >> Mute(**stage_kwargs(preset, "muting"))
        >> Filter(**stage_kwargs(preset, "filtering"))
        >> Slice(**stage_kwargs(preset, "slicing"))
        >> Selection(**stage_kwargs(preset, "selection"), flip_negatives=True)
        >> Whiten(**stage_kwargs(preset, "whitening"))
        >> Normalize(**stage_kwargs(preset, "normalization"))
        >> Apodize(method="hanning", frac=0.1)
        >> Correlate(method="cross", virtual_source_index=0, part="causal")
        >> Stack(**stage_kwargs(preset, "stacking"))
        >> Plot(folder_path=output_folder)
        >> Save(folder_path=output_folder)
        >> Pad(n=1_000, taper=25)
        >> Dispersion(method="phase", **stage_kwargs(preset, "dispersion"))
        >> Plot(folder_path=output_folder, normalize=True)
        >> Save(folder_path=output_folder)
    )
