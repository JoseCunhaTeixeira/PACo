"""The QC configuration: every threshold and budget in one place (rule 9), locked during a run
and recorded with it. Loaded from the JSON file `PACO_QC_CONFIG` names, else PACo's defaults."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sigpipe.algorithms.picking.dispersion.tracking import PickingParameters
from sigpipe.masw.inversion.priors import PriorRules
from sigpipe.masw.runs.writing import write_atomic

from paco.qc.coherence import CoherenceRules
from paco.qc.g1_signal import SignalThresholds
from paco.qc.g2_image import ImageThresholds
from paco.qc.g3_curve import CurveThresholds
from paco.qc.g4_profile import ProfileThresholds
from paco.qc.g5_model import ModelThresholds
from paco.qc.g6_models import ModelProfileThresholds
from paco.qc.g7_petro import PetroThresholds
from paco.qc.g8_petro_line import PetroLineThresholds
from paco.qc.line_loop import LineRules
from paco.qc.models import Budgets
from paco.qc.muting import MuteRules
from paco.qc.segments import SegmentRules

CONFIG_FILE = "qc_config.json"  # the snapshot in a run folder

# Retired thresholds, which an older snapshot may hold: dropped when it is read, so that its run
# still reads. The mutes' width of G2 and G3: a mute keeps each record's own pulse (G1's measure,
# shots.py), G1's own width the fallback. One SNR limit for every signal, a window's
# correlations' too (signal.min_snr_db). G3's air-wave mute, which parts nothing; G3's
# uncertainty limit, which the picker's own cap keeps from ever failing: reported, not judged.
RETIRED = {
    "image": ("mute_width_s", "min_virtual_shot_snr_db"),
    "curve": ("mute_width_s", "air_wave_mute_vmax", "max_uncertainty"),
    "signal": ("min_correlation_snr_db",),
    # The segment trials keep G2's floor on the fk selection's share (min_fk_kept_share).
    "segments": ("min_kept_share",),
}
# Thresholds renamed, their value kept under the new name (one name, one meaning):
# the first breaks' shot time off the trigger is an error, the trigger's shift the muting's.
RENAMED = {
    "signal": {"max_trigger_shift_s": "max_trigger_error_s"},
    # G5's floor on a layer's Vs over the one above, not the priors' drop (their max_vs_drop).
    "model": {"max_vs_drop": "min_vs_ratio"},
    # G6 compares the models down to their depth informed, G5's (one depth informed).
    "models": {"max_useful_depth_spread": "max_depth_informed_spread"},
}


class QCConfig(BaseModel):
    """The gates' thresholds and the retry budgets: one model per gate, with values measured on
    the demo profiles."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    budgets: Budgets = Budgets()
    coherence: CoherenceRules = Field(default_factory=CoherenceRules)  # the rules for S2
    segments: SegmentRules = Field(default_factory=SegmentRules)  # S2's, a passive line
    mute: MuteRules = Field(default_factory=MuteRules)  # the mute trial, before S1
    line: LineRules = Field(default_factory=LineRules)  # the line loop, after S2
    signal: SignalThresholds = Field(default_factory=SignalThresholds)  # G1
    image: ImageThresholds = Field(default_factory=ImageThresholds)  # G2
    curve: CurveThresholds = Field(default_factory=CurveThresholds)  # G3
    profile: ProfileThresholds = Field(default_factory=ProfileThresholds)  # G4
    priors: PriorRules = Field(default_factory=PriorRules)  # the checks before S4
    model: ModelThresholds = Field(default_factory=ModelThresholds)  # G5
    models: ModelProfileThresholds = Field(default_factory=ModelProfileThresholds)  # G6
    petro: PetroThresholds = Field(default_factory=PetroThresholds)  # G7
    petro_line: PetroLineThresholds = Field(default_factory=PetroLineThresholds)  # G8
    # Where a run's picking starts: not thresholds (the loop changes the picking), but the
    # values a run begins with. The pick goes as far as its ridge holds.
    picking: PickingParameters = Field(default_factory=PickingParameters)

    @model_validator(mode="before")
    @classmethod
    def _without_retired(cls, data: Any) -> Any:  # noqa: ANN401
        if not isinstance(data, dict):
            return data
        values = dict(data)  # pyright: ignore[reportUnknownArgumentType]
        for section, keys in RETIRED.items():
            given = values.get(section)
            if isinstance(given, dict):
                values[section] = {
                    key: value
                    for key, value in given.items()  # pyright: ignore[reportUnknownVariableType]
                    if key not in keys
                }
        for section, names in RENAMED.items():
            given = values.get(section)
            if isinstance(given, dict):
                values[section] = {
                    names.get(key, key): value  # pyright: ignore[reportUnknownArgumentType]
                    for key, value in given.items()  # pyright: ignore[reportUnknownVariableType]
                }
        return values


def load_qc_config(path: Path | None) -> QCConfig:
    """The configuration in `path`, a JSON file, or the defaults when there is none."""
    if path is None:
        return QCConfig()
    return QCConfig.model_validate_json(path.read_text())


def snapshot_qc_config(config: QCConfig, run_folder: Path) -> Path:
    """Record the configuration a run used, next to its results."""
    path = run_folder / CONFIG_FILE
    write_atomic(path, config.model_dump_json(indent=2))
    return path


def read_qc_config(run_folder: Path) -> QCConfig:
    return QCConfig.model_validate_json((run_folder / CONFIG_FILE).read_text())


def run_qc_config(run_folder: Path, path: Path | None) -> QCConfig:
    """The configuration the run's checks use: its own, or for a run the assistant has not
    checked yet (one made in PAC's pages), the one in `path` (load_qc_config), recorded next to
    its results."""
    if (run_folder / CONFIG_FILE).exists():
        return read_qc_config(run_folder)
    config = load_qc_config(path)
    snapshot_qc_config(config, run_folder)
    return config
