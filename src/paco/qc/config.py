"""The QC configuration: every threshold and budget in one place (rule 9), locked during a run
and recorded with it. Loaded from the JSON file `PACO_QC_CONFIG` names, else PACo's defaults."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
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

CONFIG_FILE = "qc_config.json"  # the snapshot in a run folder


class QCConfig(BaseModel):
    """The gates' thresholds and the retry budgets: one model per gate, with values measured on
    the demo profiles."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    budgets: Budgets = Budgets()
    coherence: CoherenceRules = Field(default_factory=CoherenceRules)  # the rules for S2
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
