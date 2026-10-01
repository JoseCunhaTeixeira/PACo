"""The settings the user gave, locked for the run: no gate or rule changes them; a gate that
needs one changed rejects its window, saying the change it asks."""

import json
from pathlib import Path

from paco.qc import Budgets, GateResult, QCReport, UnitReport, left_out
from paco.qc.budgets import budget_spent
from paco.qc.coherence import cap_band
from paco.qc.given import give, given_of, leaves, locked, said, unlocked
from paco.qc.log import LOG_FILE, read_attempts
from paco.qc.loops import RetryBudget, next_try, refusal
from paco.qc.models import Flag, Override


def _asking(stage: str, overrides: dict[str, object]) -> GateResult:
    flag = Flag(
        name="ridge_at_vmax",
        message="The ridge reaches vmax.",
        stage=stage,  # pyright: ignore[reportArgumentType]
        action=Override(stage=stage, overrides=overrides),  # pyright: ignore[reportArgumentType]
    )
    return GateResult(gate="G2", unit="xmid_2.88", verdict="retry", flags=(flag,))


def test_the_given_settings_are_kept_by_family(tmp_path: Path) -> None:
    give(tmp_path, "processing", {"dispersion": {"vmax": 400}, "mode": "active"})
    give(tmp_path, "processing", {"dispersion": {"vmin": 50}, "masw": {"length": 24}})
    give(tmp_path, "picking", {"threshold": 0.4})

    assert given_of(tmp_path) == {
        "processing": {"dispersion": {"vmax": 400, "vmin": 50}, "masw": {"length": 24}},
        "picking": {"threshold": 0.4},
    }
    assert locked(tmp_path, "phase_shift") == locked(tmp_path, "preprocessing")
    assert locked(tmp_path, "inversion") == {}
    # Events of the run's log, which the attempts' readers pass by (S2).
    lines = [json.loads(line) for line in (tmp_path / LOG_FILE).read_text().splitlines()]
    assert [(line["event"], line["family"], line["actor"]) for line in lines] == [
        ("given", "processing", "agent"),
        ("given", "processing", "agent"),
        ("given", "picking", "agent"),
    ]
    assert read_attempts(tmp_path) == ()
    assert leaves(given_of(tmp_path)["processing"]) == [
        "dispersion vmax 400",
        "dispersion vmin 50",
        "masw length 24",
    ]


def test_a_gates_change_of_a_given_setting_is_held_back() -> None:
    given = {"dispersion": {"vmax": 400}}

    free, held = unlocked({"dispersion": {"vmax": 900, "nv": 2000}}, given)

    assert (free, held) == ({"dispersion": {"nv": 2000}}, {"dispersion": {"vmax": 900}})
    assert said(held, given) == "dispersion vmax 900 (given: 400)"


def test_a_retry_that_only_changes_a_given_setting_is_refused_as_locked() -> None:
    given = {"dispersion": {"vmax": 400}}
    result = _asking("phase_shift", {"dispersion": {"vmax": 900}})
    budget = RetryBudget((), Budgets(), 4)

    assert next_try(result, "phase_shift", budget, {}, given) is None
    assert refusal(result, "phase_shift", {}, given) == (
        "locked",
        "dispersion vmax 900 (given: 400)",
    )
    rejected = budget_spent(result, "locked", "dispersion vmax 900 (given: 400)")
    assert rejected.verdict == "reject" and rejected.flags[0].name == "locked"
    assert "dispersion vmax 900 (given: 400), which the user gave" in rejected.flags[0].message
    # The window's line in the answer says the change asked, for the user to choose.
    unit = UnitReport(
        unit="xmid_2.88",
        xmid=2.88,
        verdicts={"G2": "reject"},
        flags={"G2": rejected.flags},
        attempts=1,
        parameters={},
        rejected_for=(),
        curve=None,
    )
    report = QCReport(run_id="r", budgets=Budgets(), n_xmids=1, retries=0, units=(unit,), counts={})
    assert left_out(report, ("G2",)) == (
        "xmid 2.88 (1): G2 locked, asks dispersion vmax 900 (given: 400), ridge_at_vmax",
    )
    # Without the lock, the gate's change goes through.
    assert next_try(result, "phase_shift", budget, {}, {}) is not None


def test_a_given_band_edge_stays_as_given() -> None:
    from sigpipe.masw.presets import make_preset

    preset = make_preset("active", {"dispersion": {"fmax": 120.0}})

    capped, _ = cap_band(preset, [(5.0, 80.0)], 1000.0)
    kept, kept_notes = cap_band(preset, [(5.0, 80.0)], 1000.0, {"fmax": 120.0})

    assert capped["dispersion"]["fmax"] == 80.0
    assert "fmax" not in kept.get("dispersion", {})
    assert kept_notes[0].startswith("dispersion fmax 120 Hz, as given, lies outside")


def test_a_flag_touching_a_given_setting_is_held_back_whole() -> None:
    # G5's longer sampling: the burn-in alone, with the iterations given, would sample no longer.
    longer: dict[str, object] = {"n_iterations": 4000, "n_burnin_iterations": 1000}
    result = _asking("inversion", longer)
    given = {"n_iterations": 2000}
    budget = RetryBudget((), Budgets(), 4)

    assert next_try(result, "inversion", budget, {}, given) is None
    assert refusal(result, "inversion", {}, given) == ("locked", "n_iterations 4000 (given: 2000)")
