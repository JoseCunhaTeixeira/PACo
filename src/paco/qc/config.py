"""The QC configuration: every threshold and budget in one place (rule 9), locked during a run
and recorded with it. Loaded from the JSON file `PACO_QC_CONFIG` names, else PACo's defaults."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sigpipe.algorithms.picking.dispersion.tracking import PickingParameters
from sigpipe.masw.inversion.priors import PriorRules

from paco.qc.coherence import CoherenceRules
from paco.qc.g1_signal import SignalThresholds
from paco.qc.g2_image import ImageThresholds
from paco.qc.g3_curve import CurveThresholds
from paco.qc.g4_profile import ProfileThresholds
from paco.qc.g5_model import ModelThresholds
from paco.qc.g6_models import ModelProfileThresholds
from paco.qc.g7_petro import PetroThresholds
from paco.qc.g8_petro_line import PetroLineThresholds
from paco.qc.models import Budgets
from paco.qc.segments import SegmentRules

CONFIG_FILE = "qc_config.json"  # the snapshot in a run folder

# Thresholds PACo no longer has, which a snapshot taken before may hold: dropped when it is read,
# so that its run still reads. The mutes' width of G2 and G3: each record's pulse since
# 2026-09-28 (G1's measure, shots.py), G1's own width left as the fallback.
RETIRED = {"image": ("mute_width_s",), "curve": ("mute_width_s",)}


class QCConfig(BaseModel):
    """The gates' thresholds and the retry budgets: one model per gate, with values measured on
    the demo profiles."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    budgets: Budgets = Budgets()
    coherence: CoherenceRules = Field(default_factory=CoherenceRules)  # the rules for S2
    segments: SegmentRules = Field(default_factory=SegmentRules)  # S2's, a passive line
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
        return values


def load_qc_config(path: Path | None) -> QCConfig:
    """The configuration in `path`, a JSON file, or the defaults when there is none."""
    if path is None:
        return QCConfig()
    return QCConfig.model_validate_json(path.read_text())


def snapshot_qc_config(config: QCConfig, run_folder: Path) -> Path:
    """Record the configuration a run used, next to its results."""
    path = run_folder / CONFIG_FILE
    path.write_text(config.model_dump_json(indent=2))
    return path


def read_qc_config(run_folder: Path) -> QCConfig:
    return QCConfig.model_validate_json((run_folder / CONFIG_FILE).read_text())
