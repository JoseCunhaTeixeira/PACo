"""Each record's mute width: its own pulse as G1 measured it, else the line's median, else G1's
fallback (the user, 2026-09-28)."""

from datetime import UTC, datetime

from paco.qc.config import QCConfig
from paco.qc.models import Attempt, GateResult, Metric
from paco.qc.shots import pulse_widths, with_pulse

MUTE = {"muting": {"method": "mute", "vmin": 80.0, "vmax": 1500.0}}


def _judged(unit: str, pulse: float | None) -> Attempt:
    """A record's preprocessing, G1's result on it measuring `pulse` (None: not measured)."""
    metric = Metric(name="pulse_s", value=pulse, passed=True, unit="s")
    return Attempt(
        unit=unit,
        stage="preprocessing",
        attempt=1,
        parameters={},
        triggered_by="initial",
        started_at=datetime(2026, 9, 28, tzinfo=UTC),
        status="succeeded",
        results={"G1": GateResult(gate="G1", unit=unit, verdict="pass", metrics=(metric,))},
    )


def test_each_record_keeps_its_own_pulse_else_the_lines() -> None:
    attempts = [_judged("1.dat", 0.08), _judged("2.dat", None), _judged("3.dat", 0.12)]

    widths = pulse_widths(attempts, ["1.dat", "2.dat", "3.dat"], fallback=0.05)

    # 2.dat, too noisy to measure: the line's median, of 0.08 and 0.12.
    assert widths == {"1.dat": 0.08, "2.dat": 0.1, "3.dat": 0.12}
    # No record measured: the fallback.
    assert pulse_widths([_judged("1.dat", None)], ["1.dat"], fallback=0.05) == {"1.dat": 0.05}


def test_a_mute_without_a_width_takes_the_pulse_a_given_one_stays() -> None:
    assert with_pulse(MUTE, 0.08123)["muting"]["width"] == 0.0812
    given = {"muting": {**MUTE["muting"], "width": 0.2}}
    assert with_pulse(given, 0.08) == given
    # Nothing muted: nothing to fill.
    assert with_pulse({"filtering": {"method": "iir"}}, 0.08) == {"filtering": {"method": "iir"}}


def test_a_config_saved_with_the_retired_mute_widths_still_reads() -> None:
    saved = QCConfig().model_dump(mode="json")
    saved["image"]["mute_width_s"] = 0.05  # G2's, before each record's pulse (2026-09-28)
    saved["curve"]["mute_width_s"] = 0.05

    assert QCConfig.model_validate(saved) == QCConfig()
