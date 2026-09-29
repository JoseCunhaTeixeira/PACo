import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError
from sigpipe.algorithms.picking.dispersion.tracking import PickingParameters
from sigpipe.masw.picks import CURVES_FILE
from sigpipe.masw.presets import PresetError, apply_overrides, make_preset
from sigpipe.masw.runs import RunError, find_run, load_image, run_processing

from paco.qc import (
    ATTEMPTS_FOLDER,
    CONFIG_FILE,
    LOG_FILE,
    REPORT_FILE,
    STAGE_FILES,
    Action,
    Attempt,
    Budgets,
    ExcludeRecord,
    ExcludeTraces,
    Flag,
    GateResult,
    Keep,
    Kept,
    Metric,
    Override,
    QCConfig,
    Reject,
    Stage,
    afresh,
    append_attempt,
    archived_attempts,
    attempts_of,
    budget_spent,
    build_report,
    can_retry,
    changed_settings,
    check_budget,
    describe,
    downstream,
    invalidate,
    judge_run,
    latest,
    load_qc_config,
    read_attempts,
    read_qc_config,
    read_report,
    rerun_phase_shift,
    rerun_picking,
    retries_at_gate,
    retries_by_unit,
    retries_in_run,
    run_budget,
    snapshot_qc_config,
    stage_index,
    stretches,
    summarize_report,
    write_report,
    xmid_of,
)
from paco.qc.g3_curve import CurveThresholds
from paco.qc.loops import RetryBudget, next_try, unchanged
from paco.settings import Settings

SMALL_WINDOWS = {"masw": {"length": 24, "step": 24}}

WHEN = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)

RIDGE_AT_VMAX = Flag(
    name="ridge_at_vmax",
    message="The energy maximum sits at vmax: the velocity range is too narrow.",
    stage="phase_shift",
    action=Override(stage="phase_shift", overrides={"dispersion": {"vmax": 1500}}),
)
DEAD_TRACE = Flag(
    name="dead_trace",
    message="Trace 7 is dead.",
    stage="preprocessing",
    action=ExcludeRecord(record="2.dat"),
    fixable=False,
)


def _attempt(
    unit: str,
    stage: str,
    attempt: int,
    triggered_by: str = "initial",
    result: GateResult | None = None,
) -> Attempt:
    return Attempt(
        unit=unit,
        stage=stage,  # pyright: ignore[reportArgumentType]
        attempt=attempt,
        parameters={},
        triggered_by=triggered_by,
        started_at=WHEN,
        finished_at=WHEN,
        status="succeeded",
        results={} if result is None else {result.gate: result},
    )


# ---------------------------------------------------------------- one language for every gate


def test_stages_are_in_pipeline_order() -> None:
    assert stage_index("preprocessing") < stage_index("phase_shift") < stage_index("picking")
    assert stage_index("inversion") == 3


def test_actions_round_trip_through_json_by_kind() -> None:
    actions = TypeAdapter(list[Action])
    given: list[Action] = [
        Override(stage="picking", overrides={"corridor": 0.2}),
        ExcludeRecord(record="2.dat"),
        Reject(reason="budget spent"),
        Keep(note="geology"),
    ]

    assert actions.validate_json(actions.dump_json(given)) == given


def test_a_gate_result_records_each_metric_with_its_threshold() -> None:
    result = GateResult(
        gate="G2",
        unit="xmid_12.50",
        verdict="retry",
        metrics=(
            Metric(name="coherent_columns", value=0.4, threshold=0.6, bound="min", passed=False),
        ),
        flags=(RIDGE_AT_VMAX,),
        kept=Kept(band_hz=(5.0, 60.0), n_traces=24),
    )

    assert GateResult.model_validate_json(result.model_dump_json()) == result
    with pytest.raises(ValidationError):
        GateResult(gate="G2", unit="xmid_12.50", verdict="maybe")  # pyright: ignore[reportArgumentType]


# ---------------------------------------------------------------- the QC log


def test_the_log_is_appended_and_read_back_in_order(tmp_path: Path) -> None:
    first = _attempt("xmid_12.50", "phase_shift", 1)
    second = _attempt("xmid_12.50", "phase_shift", 2, "G2:ridge_at_vmax")

    assert read_attempts(tmp_path) == ()
    append_attempt(tmp_path, first)
    append_attempt(tmp_path, second)

    assert read_attempts(tmp_path) == (first, second)
    assert len((tmp_path / LOG_FILE).read_text().splitlines()) == 2
    assert latest(read_attempts(tmp_path), "xmid_12.50", "phase_shift") == second
    assert latest(read_attempts(tmp_path), "xmid_12.50", "picking") is None


def test_retries_are_counted_from_what_triggered_each_attempt() -> None:
    attempts = [
        _attempt("xmid_12.50", "phase_shift", 1),
        _attempt("xmid_12.50", "phase_shift", 2, "G2:ridge_at_vmax"),
        _attempt("xmid_12.50", "picking", 1),
        _attempt("xmid_12.50", "picking", 2, "G3:air_wave"),
        _attempt("xmid_13.00", "phase_shift", 1),
        _attempt("xmid_13.00", "phase_shift", 2, "G3:mode_jump"),  # G3 sent it back to S2
    ]

    assert retries_at_gate(attempts, "xmid_12.50", "G2") == 1
    assert retries_at_gate(attempts, "xmid_12.50", "G3") == 1
    assert retries_at_gate(attempts, "xmid_13.00", "G3") == 1
    assert retries_in_run(attempts) == 3
    assert retries_by_unit(attempts) == {"xmid_12.50": 2, "xmid_13.00": 1}


def test_a_stage_started_afresh_forgets_its_attempts_not_their_cost(tmp_path: Path) -> None:
    for one in (
        _attempt("xmid_12.50", "phase_shift", 1),
        _attempt("xmid_12.50", "phase_shift", 2, "G2:ridge_at_vmax"),
        _attempt("xmid_12.50", "picking", 1),
        _attempt("xmid_13.00", "phase_shift", 1),
    ):
        append_attempt(tmp_path, one)
    write_report(build_report("run", tmp_path, Budgets(), 2), tmp_path)
    old = (tmp_path / REPORT_FILE).stat().st_mtime - 10
    os.utime(tmp_path / REPORT_FILE, (old, old))

    # The agent asked to do xmid_12.50's phase shift again, a wider range: its attempts at it
    # and at the stages after it forgotten, what they cost and the last one's parameters carried
    # by the new one.
    wider = _attempt("xmid_12.50", "phase_shift", 3, "backtrack").model_copy(
        update={"parameters": {"dispersion": {"vmax": 900}}}
    )
    again = afresh(tmp_path, wider)
    append_attempt(tmp_path, again)

    assert (again.attempt, again.forgotten, again.replaced) == (1, 1, {})
    attempts = read_attempts(tmp_path)
    assert [(a.unit, a.stage, a.attempt) for a in attempts] == [
        ("xmid_13.00", "phase_shift", 1),
        ("xmid_12.50", "phase_shift", 1),
    ]
    # The G2 retry's cost kept, the backtrack's added: the run's budget never refunded.
    assert retries_in_run(attempts) == 2
    # The report, older than the log now: built again from it, the change read against the
    # attempt replaced.
    report = read_report(tmp_path)
    assert report.retries == 2
    assert [unit.unit for unit in report.units] == ["xmid_13.00", "xmid_12.50"]
    assert report.changed["backtrack"] == ("phase_shift", {"dispersion": {"vmax": 900}}, True)


# ---------------------------------------------------------------- budgets


def test_budgets_have_their_default_values() -> None:
    assert Budgets() == Budgets(per_gate_and_unit=2, per_xmid_of_the_run=2)
    assert run_budget(Budgets(), n_xmids=73) == 146


def test_a_unit_may_retry_until_either_budget_is_spent() -> None:
    budgets = Budgets(per_gate_and_unit=1, per_xmid_of_the_run=1)
    none_yet = [_attempt("xmid_12.50", "phase_shift", 1)]
    gate_spent = [*none_yet, _attempt("xmid_12.50", "phase_shift", 2, "G2:ridge_at_vmax")]
    # Two xmids allow two retries in the run; the other xmid took them both.
    run_spent = [
        *none_yet,
        _attempt("xmid_13.00", "picking", 2, "G3:air_wave"),
        _attempt("xmid_13.00", "picking", 3, "G3:air_wave"),
    ]

    assert can_retry(none_yet, budgets, 2, "xmid_12.50", "G2")
    assert not can_retry(gate_spent, budgets, 2, "xmid_12.50", "G2")
    assert can_retry(gate_spent, budgets, 2, "xmid_12.50", "G3")  # another gate
    assert not can_retry(run_spent, budgets, 2, "xmid_12.50", "G2")


def test_a_spent_budget_rejects_with_the_last_flags() -> None:
    asked = GateResult(gate="G2", unit="xmid_12.50", verdict="retry", flags=(RIDGE_AT_VMAX,))

    rejected = budget_spent(asked)

    assert rejected.verdict == "reject"
    assert [flag.name for flag in rejected.flags] == ["budget_spent", "ridge_at_vmax"]
    assert rejected.flags[0].action == Reject(reason="budget spent")
    assert not rejected.flags[0].fixable
    assert "ridge_at_vmax" in rejected.flags[0].message


def test_a_retry_that_changes_nothing_is_refused_and_said() -> None:
    # Asked with the parameters of the attempt before, it would give its result again (the user,
    # 2026-09-29): none granted, the unit rejected with why.
    asked = GateResult(gate="G2", unit="xmid_12.50", verdict="retry", flags=(RIDGE_AT_VMAX,))
    changes = RIDGE_AT_VMAX.action.model_dump()["overrides"]
    budget = RetryBudget((), Budgets(), 2)

    assert next_try(asked, "phase_shift", budget, changes) is None
    assert unchanged(asked, "phase_shift", changes) and not unchanged(asked, "phase_shift", {})
    assert next_try(asked, "phase_shift", budget, {}) is not None
    said = {why: budget_spent(asked, why).flags[0] for why in ("budget", "unchanged", "redone")}
    assert [flag.name for flag in said.values()] == [
        "budget_spent",
        "nothing_to_try",
        "redone_once",
    ]
    assert all(not flag.fixable and "ridge_at_vmax" in flag.message for flag in said.values())


def test_g1s_retries_draw_on_the_records_budgets_alone() -> None:
    # The run's budget spent by the windows: a record may still have its trigger corrected.
    started = datetime(2026, 9, 25, 12, 0, tzinfo=UTC)
    spent = tuple(
        Attempt(
            unit="xmid_2.88",
            stage="phase_shift",
            attempt=attempt,
            parameters={},
            triggered_by="G2:ridge_at_vmax",
            started_at=started,
            status="succeeded",
        )
        for attempt in (2, 3, 4, 5)
    )
    budget = RetryBudget(spent, Budgets(), 2)

    assert budget.left == 0
    assert budget.grant("1.dat", "G1") and not budget.grant("xmid_8.88", "G2")


# ---------------------------------------------------------------- the configuration


def test_a_snapshot_with_retired_or_renamed_thresholds_still_reads(tmp_path: Path) -> None:
    # A run judged before 2026-09-29: its thresholds since retired dropped, those renamed read
    # under their new name (one name, one meaning).
    given = tmp_path / "qc.json"
    given.write_text(
        '{"signal": {"min_correlation_snr_db": 3.0, "max_trigger_shift_s": 0.02},'
        ' "image": {"min_virtual_shot_snr_db": 3.0},'
        ' "curve": {"air_wave_mute_vmax": 320.0},'
        ' "model": {"max_vs_drop": 0.4},'
        ' "models": {"max_useful_depth_spread": 0.6}}'
    )

    config = load_qc_config(given)

    assert config.signal.max_trigger_error_s == 0.02
    assert config.model.min_vs_ratio == 0.4
    assert config.models.max_depth_informed_spread == 0.6


def test_g1s_retries_are_the_records_not_the_windows() -> None:
    started = datetime(2026, 9, 25, 12, 0, tzinfo=UTC)

    def attempt(unit: str, stage: Stage, trigger: str) -> Attempt:
        return Attempt(
            unit=unit,
            stage=stage,
            attempt=2,
            parameters={},
            triggered_by=trigger,
            started_at=started,
            status="succeeded",
        )

    # Each record's trigger corrected: bounded by the record's own budget, not the run's.
    attempts = (
        attempt("1.dat", "preprocessing", "G1:shifted_trigger"),
        attempt("2.dat", "preprocessing", "G1:shifted_trigger"),
        attempt("xmid_2.88", "phase_shift", "G2:ridge_at_vmax"),
        attempt("2.dat", "preprocessing", "backtrack"),  # the agent's: on the run's budget
    )
    assert retries_in_run(attempts) == 2
    # Two windows: 4 retries. G1's counted, they would all be spent before any window's.
    assert can_retry(attempts, Budgets(), 2, "xmid_8.88", "G2")
    more = (*attempts, attempt("xmid_8.88", "phase_shift", "G2:ridge_at_vmax"))
    assert not can_retry((*more, more[-1]), Budgets(), 2, "xmid_14.88", "G2")


def test_going_back_is_refused_once_the_runs_budget_is_spent(tmp_path: Path) -> None:
    started = datetime(2026, 9, 25, 12, 0, tzinfo=UTC)
    for attempt in range(2, 6):  # four retries: the budget of a run of two windows
        append_attempt(
            tmp_path,
            Attempt(
                unit="xmid_1.00",
                stage="picking",
                attempt=attempt,
                parameters={},
                triggered_by="backtrack",
                started_at=started,
                status="succeeded",
            ),
        )

    with pytest.raises(RunError, match=r"The retry budget of run 'r' is spent \(4 of 4\): you"):
        check_budget("r", tmp_path, QCConfig(), 2)
    check_budget("r", tmp_path, QCConfig(), 3)  # 4 of 6: not spent


def test_the_configuration_defaults_hold_the_thresholds_and_the_budgets() -> None:
    config = QCConfig()

    assert config.budgets == Budgets()
    assert config.curve == CurveThresholds()
    # The run's picking starts there: as far as the ridge holds.
    assert config.picking == PickingParameters()
    assert load_qc_config(None) == config


def test_the_configuration_is_loaded_from_a_file_and_recorded_with_the_run(tmp_path: Path) -> None:
    given = tmp_path / "qc.json"
    given.write_text(
        '{"budgets": {"per_gate_and_unit": 3}, "curve": {"metrics": {"min_sharpness": 0.9}}}'
    )
    run_folder = tmp_path / "run"
    run_folder.mkdir()

    config = load_qc_config(given)
    snapshot_qc_config(config, run_folder)

    assert config.budgets.per_gate_and_unit == 3
    assert config.budgets.per_xmid_of_the_run == 2
    assert config.curve.metrics.min_sharpness == 0.9
    assert read_qc_config(run_folder) == config
    # A typo in a threshold's name is refused, like the tools' arguments.
    given.write_text('{"curve": {"metrics": {"min_sharpnes": 1}}}')
    with pytest.raises(ValidationError, match="min_sharpnes"):
        load_qc_config(given)


# ---------------------------------------------------------------- attempts kept inside one run


def test_going_back_to_a_stage_archives_it_and_everything_after(tmp_path: Path) -> None:
    window = tmp_path / "xmid_12.50"
    window.mkdir()
    files = {
        "window.json": "phase_shift keeps it",
        "Stream_0000.png": "S2",
        "DispersionImage_0000.hdf5": "S2",
        "DispersionImage_0000.png": "S2, redrawn by S3",
        "DispersionCurves_0000.csv": "S3",
        "quality.json": "S3",
        "SeismicInversion_Model_0000_median.csv": "S4",
        "PetroInversion_Model_0000.csv": "the petrophysical inversion, from the picks too",
    }
    for name, content in files.items():
        (window / name).write_text(content)

    archive = invalidate(window, "picking", attempt=1)

    assert archive == window / ATTEMPTS_FOLDER / "1_picking"
    assert {path.name for path in archive.iterdir()} == {
        "DispersionCurves_0000.csv",
        "quality.json",
        "SeismicInversion_Model_0000_median.csv",
        "PetroInversion_Model_0000.csv",
    }
    # The image stays for the picking to run again; window.json is the window itself.
    assert {path.name for path in window.iterdir() if path.is_file()} == {
        "window.json",
        "Stream_0000.png",
        "DispersionImage_0000.hdf5",
        "DispersionImage_0000.png",
    }
    assert (archive / "quality.json").read_text() == "S3"
    assert archived_attempts(window) == (archive,)
    assert downstream("phase_shift") == ("phase_shift", "picking", "inversion", "petro_inversion")
    assert STAGE_FILES["preprocessing"] == ()


# ---------------------------------------------------------------- the report


def _result(
    gate: str, unit: str, verdict: str, *flags: Flag, kept: Kept | None = None
) -> GateResult:
    return GateResult(
        gate=gate,
        unit=unit,
        verdict=verdict,  # pyright: ignore[reportArgumentType]
        flags=flags,
        kept=kept or Kept(),
    )


def test_the_report_gives_each_unit_its_verdicts_attempts_parameters_and_curve(
    tmp_path: Path,
) -> None:
    curve = Kept(band_hz=(8.0, 45.0), wavelength_m=(2.0, 20.0), n_points=12)
    for attempt in [
        _attempt("1.dat", "preprocessing", 1, result=_result("G1", "1.dat", "pass")),
        _attempt(
            "xmid_12.50",
            "phase_shift",
            1,
            result=_result("G2", "xmid_12.50", "retry", RIDGE_AT_VMAX),
        ),
        Attempt(
            unit="xmid_12.50",
            stage="phase_shift",
            attempt=2,
            parameters={"dispersion": {"vmax": 1500}},
            triggered_by="G2:ridge_at_vmax",
            started_at=WHEN,
            finished_at=WHEN,
            status="succeeded",
            results={"G2": _result("G2", "xmid_12.50", "pass")},
        ),
        _attempt(
            "xmid_12.50", "picking", 1, result=_result("G3", "xmid_12.50", "pass", kept=curve)
        ),
        _attempt(
            "xmid_13.00", "phase_shift", 1, result=_result("G2", "xmid_13.00", "reject", DEAD_TRACE)
        ),
    ]:
        append_attempt(tmp_path, attempt)

    report = build_report("20260924-120000-abcd", tmp_path, Budgets(), n_xmids=2)
    write_report(report, tmp_path)

    assert read_report(tmp_path) == report
    assert report.retries == 1
    assert report.counts == {"G1": {"pass": 1}, "G2": {"pass": 1, "reject": 1}, "G3": {"pass": 1}}
    record, first, second = report.units
    assert (record.unit, record.xmid, record.verdicts) == ("1.dat", None, {"G1": "pass"})
    assert first.xmid == 12.5
    assert first.verdicts == {"G2": "pass", "G3": "pass"}
    assert first.attempts == 3
    assert first.parameters == {"phase_shift": {"dispersion": {"vmax": 1500}}, "picking": {}}
    assert first.curve == curve
    assert first.rejected_for == ()
    assert second.rejected_for == ("Trace 7 is dead.",)
    assert report.rejected == (second,)


def test_the_summary_groups_the_flags_by_stretches_of_xmids(tmp_path: Path) -> None:
    for xmid in (12.0, 12.5, 13.0, 14.0):
        unit = f"xmid_{xmid:.2f}"
        append_attempt(
            tmp_path,
            _attempt(unit, "phase_shift", 1, result=_result("G2", unit, "retry", RIDGE_AT_VMAX)),
        )
    append_attempt(
        tmp_path,
        _attempt("2.dat", "preprocessing", 1, result=_result("G1", "2.dat", "reject", DEAD_TRACE)),
    )

    text = summarize_report(build_report("20260924-120000-abcd", tmp_path, Budgets(), n_xmids=4))

    assert text.splitlines() == [
        "G2: 4 retry",
        "G1: 1 reject",
        "Retries: 0 of 8.",
        "G2 ridge_at_vmax, xmid 12.00-13.00 (3), 14.00 (1): The energy maximum sits at vmax: the "
        'velocity range is too narrow. -> phase_shift {"dispersion":{"vmax":1500}}',
        "G1 dead_trace, 2.dat: Trace 7 is dead. -> exclude record 2.dat",
    ]


def test_the_summary_gives_the_checks_notes_and_the_failures(tmp_path: Path) -> None:
    started = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    for unit, xmid_error in (
        ("xmid_1.00", None),
        ("xmid_1.25", "ValueError: shapes"),
        ("xmid_1.50", "ValueError: shapes"),
    ):
        append_attempt(
            tmp_path,
            Attempt(
                unit=unit,
                stage="inversion",
                attempt=1,
                parameters={},
                triggered_by="initial",
                started_at=started,
                status="failed" if xmid_error else "succeeded",
                error=xmid_error,
                notes=(
                    "vs_min above the curve's slowest velocity (150 m/s) in layer 1: set to 120 m/s.",
                ),
            ),
        )
    append_attempt(
        tmp_path,
        Attempt(
            unit="line",
            stage="phase_shift",
            attempt=1,
            parameters={"masw": {"length": 24}},
            triggered_by="initial",
            started_at=started,
            status="succeeded",
            notes=("masw length 24 for the whole line.",),
        ),
    )

    report = build_report("20260924-120000-abcd", tmp_path, Budgets(), 3)
    lines = summarize_report(report).splitlines()

    assert "Changes at inversion, xmid 1.00-1.50 (3): vs_min above the curve's slowest " in (
        "\n".join(lines)
    )
    assert "Changes at phase_shift, line: masw length 24 for the whole line." in lines
    assert "Failed inversion, xmid 1.25-1.50 (2): ValueError: shapes" in lines
    assert {unit.unit: unit.failed for unit in report.units}["xmid_1.25"] == {
        "inversion": "ValueError: shapes"
    }


def test_the_checks_notes_outlive_a_retry_that_notes_nothing(tmp_path: Path) -> None:
    # The checks changed the user's value at the first inversion; G5's retry derives from values
    # already checked and notes nothing: the note stays the unit's.
    note = "vs_max 180 m/s below 1.09 times the curve's fastest velocity: set to 450 m/s."
    for attempt, notes, trigger in ((1, (note,), "initial"), (2, (), "G5:not_converged")):
        append_attempt(
            tmp_path,
            Attempt(
                unit="xmid_1.00",
                stage="inversion",
                attempt=attempt,
                parameters={"n_iterations": 1_000 * attempt},
                triggered_by=trigger,
                started_at=WHEN,
                status="succeeded",
                notes=notes,
            ),
        )

    report = build_report("20260926-050000-abcd", tmp_path, Budgets(), 1)

    (unit,) = report.units
    assert unit.notes == {"inversion": (note,)}
    assert any("180" in said for said in changed_settings(report, ("inversion",)))


def test_the_settings_the_gates_changed_are_said_from_and_to(tmp_path: Path) -> None:
    started = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)

    def log(
        unit: str,
        attempt: int,
        trigger: str,
        parameters: dict[str, object],
        notes: tuple[str, ...] = (),
    ) -> None:
        append_attempt(
            tmp_path,
            Attempt(
                unit=unit,
                stage="phase_shift",
                attempt=attempt,
                parameters=parameters,
                triggered_by=trigger,
                started_at=started,
                status="succeeded",
                notes=notes,
            ),
        )

    for unit in ("xmid_1.00", "xmid_2.00"):
        log(unit, 1, "initial", {"dispersion": {"vmax": 250.0}})
        log(unit, 2, "G2:ridge_at_vmax", {"dispersion": {"vmax": 375.0}})
    log("xmid_3.00", 1, "initial", {"dispersion": {"vmax": 250.0}})
    log("line", 1, "initial", {"masw": {"length": 32}}, ("masw length 32 for the whole line.",))

    changed = changed_settings(build_report("20260924-120000-abcd", tmp_path, Budgets(), 3))

    assert changed == (
        "dispersion vmax 250 -> 375 at xmid 1.00-2.00 (2), by G2:ridge_at_vmax",
        "line: masw length 32 for the whole line.",
    )


@pytest.mark.parametrize(
    ("xmids", "text"),
    [
        ([], "no xmid"),
        ([12.5], "xmid 12.50 (1)"),
        ([12.0, 12.5, 13.0, 14.0], "xmid 12.00-13.00 (3), 14.00 (1)"),
        ([2.88, 8.88, 14.88, 20.88], "xmid 2.88-20.88 (4)"),
    ],
)
def test_stretches(xmids: list[float], text: str) -> None:
    assert stretches(xmids) == text


def test_stretches_follow_the_lines_spacing() -> None:
    # xmids 0.25 m apart, named to 2 decimals: 0.24 and 0.26 m apart in turn.
    assert stretches([1.88, 2.12, 2.38, 2.62, 3.88], 0.24) == "xmid 1.88-2.62 (4), 3.88 (1)"
    # Two windows far apart on a dense line are not a stretch.
    assert stretches([1.88, 11.88], 0.24) == "xmid 1.88 (1), 11.88 (1)"


def test_describe_each_action() -> None:
    assert (
        describe(Override(stage="picking", overrides={"corridor": 0.2}))
        == 'picking {"corridor":0.2}'
    )
    assert (
        describe(ExcludeTraces(record="1.dat", traces=(3, 7))) == "exclude traces [3, 7] of 1.dat"
    )
    assert describe(ExcludeRecord(record="2.dat")) == "exclude record 2.dat"
    assert describe(Reject(reason="budget spent")) == "reject: budget spent"
    assert describe(Keep(note="an inverse trend is geology")) == "keep: an inverse trend is geology"
    assert (xmid_of("xmid_12.50"), xmid_of("1.dat"), xmid_of("xmid_x")) == (12.5, None, None)


# ---------------------------------------------------------------- a stage done again, on a real run


@pytest.fixture(scope="module")
def processed(
    demo_input_dir: Path, tmp_path_factory: pytest.TempPathFactory
) -> tuple[Settings, str]:
    """active_p1 processed once, in four 24-receiver windows."""
    root = tmp_path_factory.mktemp("rerun")
    settings = Settings(input_dir=demo_input_dir, output_dir=root / "outputs", workers=2)
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(root)
        summary = run_processing("active_p1", "active", SMALL_WINDOWS, settings)
    return settings, summary.run_id


def test_the_phase_shift_done_again_keeps_the_old_results_and_logs_the_attempt(
    processed: tuple[Settings, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    settings, run_id = processed
    run_folder = find_run(run_id, settings)
    before = (run_folder / "xmid_2.88" / "DispersionImage_0000.hdf5").read_bytes()

    outcomes = rerun_phase_shift(
        run_id,
        ["xmid_2.88", "xmid_8.88"],
        {"dispersion": {"vmax": 500}},
        settings,
        "G2:ridge_at_vmax",
    )

    assert [(outcome.folder, outcome.status) for outcome in outcomes] == [
        ("xmid_2.88", "succeeded"),
        ("xmid_8.88", "succeeded"),
    ]
    # The first attempt's image is kept; the new one has the new velocity range.
    archive = run_folder / "xmid_2.88" / ATTEMPTS_FOLDER / "1_phase_shift"
    assert (archive / "DispersionImage_0000.hdf5").read_bytes() == before
    assert load_image(run_folder / "xmid_2.88").vs.max() == pytest.approx(500.0)
    assert load_image(run_folder / "xmid_14.88").vs.max() == pytest.approx(1000.0)  # untouched
    # The log holds the run's first attempts, then the two new ones.
    attempts = read_attempts(run_folder)
    assert [(a.unit, a.stage, a.attempt, a.triggered_by) for a in attempts] == [
        ("1.dat", "preprocessing", 1, "initial"),
        ("2.dat", "preprocessing", 1, "initial"),
        ("xmid_2.88", "phase_shift", 1, "initial"),
        ("xmid_8.88", "phase_shift", 1, "initial"),
        ("xmid_14.88", "phase_shift", 1, "initial"),
        ("xmid_20.88", "phase_shift", 1, "initial"),
        ("xmid_2.88", "phase_shift", 2, "G2:ridge_at_vmax"),
        ("xmid_8.88", "phase_shift", 2, "G2:ridge_at_vmax"),
    ]
    assert attempts[-1].parameters == {"dispersion": {"vmax": 500}}
    assert attempts[-1].status == "succeeded"
    # Asked of the agent (a backtrack): the window's image afresh, nothing older of it left, but
    # what its retry cost the run's budget.
    spent = retries_in_run(read_attempts(run_folder))
    rerun_phase_shift(run_id, ["xmid_2.88"], {"dispersion": {"vmax": 600}}, settings)
    attempts = read_attempts(run_folder)
    own = [(a.attempt, a.triggered_by, a.forgotten) for a in attempts if a.unit == "xmid_2.88"]
    assert own == [(1, "backtrack", 1)]
    assert archived_attempts(run_folder / "xmid_2.88") == ()
    assert load_image(run_folder / "xmid_2.88").vs.max() == pytest.approx(600.0)
    assert retries_in_run(attempts) == spent + 1


def test_windows_cannot_move_and_must_exist(processed: tuple[Settings, str]) -> None:
    settings, run_id = processed

    with pytest.raises(RunError, match="masw cannot change"):
        rerun_phase_shift(run_id, ["xmid_2.88"], {"masw": {"length": 12}}, settings)
    with pytest.raises(
        RunError, match=r"no window xmid_3\.00\. Its windows: xmid_2\.88, xmid_8\.88"
    ):
        rerun_phase_shift(run_id, ["xmid_3.00"], {}, settings)


def test_apply_overrides_merges_a_stage_and_replaces_a_method() -> None:
    preset = make_preset("active", {"filtering": {"method": "iir", "fmin": 5, "fmax": 50}})

    merged = apply_overrides(preset, {"filtering": {"fmax": 60}, "dispersion": {"vmax": 500}})
    replaced = apply_overrides(preset, {"filtering": {"method": "none"}})

    assert merged.model_dump()["filtering"] == {
        "method": "iir",
        "fmin": 5.0,
        "fmax": 60.0,
        "order": 4,
    }
    assert merged.model_dump()["dispersion"]["vmax"] == 500.0
    assert replaced.model_dump()["filtering"] == {"method": "none"}
    with pytest.raises(PresetError, match=r"dispersion\.vmaxx: unknown parameter"):
        apply_overrides(preset, {"dispersion": {"vmaxx": 500}})


# ---------------------------------------------------------------- the gates on a whole run


def test_judge_run_puts_the_four_gates_in_the_log_and_the_report(
    demo_input_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A run of its own: the shared one has windows done again with other velocity ranges.
    monkeypatch.chdir(tmp_path)
    settings = Settings(input_dir=demo_input_dir, output_dir=tmp_path / "outputs", workers=2)
    run_id = run_processing("active_p1", "active", SMALL_WINDOWS, settings).run_id
    run_folder = find_run(run_id, settings)

    report = judge_run(run_id, settings, QCConfig())

    assert (run_folder / CONFIG_FILE).exists()
    assert read_report(run_folder) == report
    # G1 on the two records: their files say the shot comes 20 ms in, where the first breaks put
    # it (within 10 ms); not muted, nothing to correct (the trigger is part of the muting).
    for name in ("1.dat", "2.dat"):
        unit = next(unit for unit in report.units if unit.unit == name)
        assert "shifted_trigger" not in {flag.name for flag in unit.flags.get("G1", ())}
    # G2 and G3 on the four windows: G3 passes the four 24-receiver windows, their picks
    # followed past their ends (xmid 20.88, the line's last window, kept 2 points before
    # 2026-09-28, now 34), its kept flags said: the long wavelengths past three window lengths,
    # the shots nearer than half the longest, an inverse trend.
    windows = [unit for unit in report.units if unit.xmid is not None]
    assert all(unit.verdicts["G3"] == "pass" for unit in windows)
    assert {unit.unit: [flag.name for flag in unit.flags["G3"]] for unit in windows} == {
        "xmid_2.88": ["inverse_dispersion", "near_field"],
        "xmid_8.88": ["beyond_reach", "near_field"],
        "xmid_14.88": ["beyond_reach", "near_field"],
        "xmid_20.88": ["beyond_reach", "near_field"],
    }
    # The band reaches the image's 100 Hz: kept, not widened. The records are usable to 412.5 Hz,
    # counted within the image's frequencies (since 2026-09-29): its coherent band covers them.
    assert all(unit.verdicts["G2"] == "pass" for unit in windows)
    assert all({flag.name for flag in unit.flags["G2"]} == {"band_at_fmax"} for unit in windows)
    assert all(unit.curve is not None and unit.curve.n_points for unit in windows)
    # S3 saved each curve in PAC's layout.
    assert all((run_folder / unit.unit / CURVES_FILE).exists() for unit in windows)
    # G4 over the line: four curves 6 m apart, one or two neighbours a side, are no evidence
    # against each other; no window without a curve, no gap.
    assert all(unit.verdicts["G4"] == "pass" for unit in windows)
    line = next(unit for unit in report.units if unit.unit == "line")
    assert line.verdicts == {"G4": "pass"}
    assert line.flags["G4"] == ()
    # The picking is an attempt of its own, judged by G3.
    attempts = read_attempts(run_folder)
    assert latest(attempts, "xmid_2.88", "picking") is not None
    picked = latest(attempts, "xmid_2.88", "picking")
    assert picked is not None and "G3" in picked.results
    # The configuration's picking: as far as the ridge holds, no wavelength limit.
    assert (picked.parameters["max_gap_hz"], picked.parameters["max_wavelength"]) == (2.0, None)
    text = summarize_report(report)
    assert "G2 band_at_fmax, xmid 2.88-20.88 (4)" in text
    assert "shifted_trigger" not in text
    assert "G3 beyond_reach, xmid 8.88-20.88 (3)" in text
    assert "gaps" not in text

    # The picking again for a window, with an override (asked of the agent): the window's
    # picking afresh, its only attempt now, the first curve not kept; G3 on the new one.
    looser = {"min_relative_coherence": 0.3}
    (result,) = rerun_picking(run_id, ["xmid_20.88"], looser, settings)
    assert result.gate == "G3" and result.unit == "xmid_20.88"
    picks = attempts_of(read_attempts(run_folder), "xmid_20.88", "picking")
    assert [(one.attempt, one.triggered_by) for one in picks] == [(1, "backtrack")]
    (second,) = picks
    assert second.parameters["min_relative_coherence"] == 0.3
    assert second.parameters["max_gap_hz"] == 2.0
    assert second.results["G3"] == result
    assert archived_attempts(run_folder / "xmid_20.88") == ()
    with pytest.raises(RunError, match=r"no processed window xmid_1\.00"):
        rerun_picking(run_id, ["xmid_1.00"], {}, settings)
    with pytest.raises(RunError, match="Unknown picking parameter"):
        rerun_picking(run_id, ["xmid_2.88"], {"coridor": 0.1}, settings)
