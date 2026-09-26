"""PAC's passive-active pipeline, from the preprocessed records: interferometry on the window's
shots (each gather cut to its surface-wave window, cross-correlated with the receiver nearest
its shot, flipped when the shot is past the window's far end, and seen from its first receiver:
PACo's additions), the correlations stacked, then a dispersion image."""

from pathlib import Path

from sigpipe.base import Pipeline
from sigpipe.transformers import ActiveShotCorrelation, Apodize, Dispersion, Plot, Save, Stack

from paco.pipelines.common import load_preprocessed, stage_kwargs
from paco.presets import PassiveActivePreset
from paco.transformers import FromFirstReceiver, SurfaceWaveWindow
from paco.windows import MASWWindow


def build_passive_active_pipeline(
    preset: PassiveActivePreset, window: MASWWindow, records_folder: Path, output_folder: Path
) -> Pipeline:
    shots = load_preprocessed(window, records_folder)
    surface_waves = stage_kwargs(preset, "correlation_window")
    if surface_waves.pop("method") != "none":
        shots = shots >> SurfaceWaveWindow(**surface_waves)
    return (
        shots
        >> Apodize(method="hanning", frac=0.1)
        >> ActiveShotCorrelation(method="cross")
        >> FromFirstReceiver()
        >> Stack(**stage_kwargs(preset, "stacking"))
        >> Plot(folder_path=output_folder)
        >> Save(folder_path=output_folder)
        >> Dispersion(method="phase", **stage_kwargs(preset, "dispersion"))
        >> Plot(folder_path=output_folder, normalize=True)
        >> Save(folder_path=output_folder)
    )
