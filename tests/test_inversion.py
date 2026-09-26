import logging
import re
import shutil
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import h5py
import pytest
from pydantic import ValidationError
from sigpipe.masw.inversion import InversionError, InversionParameters, ThicknessLayer, VsLayer
from sigpipe.masw.inversion.section import (
    COMPARISON_FIGURE,
    SECTION_FIGURE,
    SECTION_FILE,
    save_comparison,
    save_section,
)
from sigpipe.masw.runs import RunError

from paco.inversion import (
    InversionRecord,
    WindowInversion,
    find_job,
    read_record,
    summarize_inversion,
    write_record,
)
from paco.jobs import JobManager
from paco.qc import QCConfig, run_inversion_job, submit_inversion
from paco.qc.curves import pick_line
from paco.qc.line import process_line
from paco.settings import Settings

# Four 24-receiver windows along the active demo line, as in test_runs.py; G3 and G4 pass the
# four curves (each pick stops where its ridge breaks).
SMALL_WINDOWS = {"masw": {"length": 24, "step": 24}}
# A short sampler: every step of an inversion, in about a second per window; far too short for
# G5, whose retries (twice the iterations, twice) are then spent.
SHORT = {"n_iterations": 500, "n_burnin_iterations": 50, "n_chains": 1}
# xmid 2.88 has no curve: 2 points once G1 leaves trace 13 out of its image (the decay fitted
# within the reach).
PICKED = ("xmid_8.88", "xmid_14.88", "xmid_20.88")
# The files PAC's invert_position writes in a window folder.
PAC_FILES = {
    "SeismicInversion_DensityCurves_0000.png",
    "SeismicInversion_DispersionCurves_0000_best.csv",
    "SeismicInversion_DispersionCurves_0000_ensemble.csv",
    "SeismicInversion_DispersionCurves_0000_median.csv",
    "SeismicInversion_DispersionCurves_0000_smooth_best.csv",
    "SeismicInversion_DispersionCurves_0000_smooth_median.csv",
    "SeismicInversion_DispersionImage_0000.png",
    "SeismicInversion_Log_0000.log",
    "SeismicInversion_Marginals_0000.png",
    "SeismicInversion_Model_0000_best.csv",
    "SeismicInversion_Model_0000_ensemble.csv",
    "SeismicInversion_Model_0000_median.csv",
    "SeismicInversion_Model_0000_smooth_best.csv",
    "SeismicInversion_Model_0000_smooth_median.csv",
}


@dataclass(frozen=True)
class Picked:
    settings: Settings
    run_id: str
    folder: Path


@dataclass(frozen=True)
class Inverted:
    record: InversionRecord
    folder: Path
    progress: list[tuple[int, int]]


# The real runs are the slow part: one run is processed and picked the QC way, and inverted
# once with a short sampler. Tests that change a run work on a copy.
@pytest.fixture(scope="module")
def picked(demo_input_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Picked:
    root = tmp_path_factory.mktemp("inversion")
    settings = Settings(input_dir=demo_input_dir, output_dir=root / "outputs", workers=2)
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(root)
        run_id = process_line("active_p1", SMALL_WINDOWS, settings, QCConfig()).run_id
        pick_line(run_id, settings)
    return Picked(settings, run_id, settings.output_dir / "active_p1" / run_id)


@pytest.fixture(scope="module")
def inverted(picked: Picked, tmp_path_factory: pytest.TempPathFactory) -> Inverted:
    settings, folder = _copy(picked, tmp_path_factory.mktemp("inverted"))
    progress: list[tuple[int, int]] = []
    record = submit_inversion(picked.run_id, SHORT, settings)
    record = run_inversion_job(record, settings, lambda done, total: progress.append((done, total)))
    return Inverted(record, folder, progress)


def _copy(picked: Picked, root: Path) -> tuple[Settings, Path]:
    """A copy of the picked run, to change."""
    settings = picked.settings.model_copy(update={"output_dir": root})
    folder = shutil.copytree(picked.folder, root / "active_p1" / picked.run_id)
    return settings, folder


def _record(**changes: object) -> InversionRecord:
    """A record of a two-window job, with `changes`."""
    fields: dict[str, object] = {
        "job_id": "inv-20260923-100000-abcd",
        "run_id": "20260923-090000-abcd",
        "given": SHORT,
        "state": "running",
        "submitted_at": datetime(2026, 9, 23, 10, 0, tzinfo=UTC),
        "started_at": datetime(2026, 9, 23, 10, 0, tzinfo=UTC),
        "total": 2,
    }
    return InversionRecord.model_validate(fields | changes)


def _window(
    xmid: float,
    vs: tuple[float, ...],
    thicknesses: tuple[float, ...],
    smooth: tuple[float, ...] = (),
    useful_depth: float | None = None,
    misfit: float | None = None,
) -> WindowInversion:
    return WindowInversion(
        xmid=xmid,
        folder=f"xmid_{xmid:.2f}",
        status="succeeded",
        vs_m_s=vs,
        thicknesses_m=thicknesses,
        vs_at_depths_m_s=smooth or None,
        useful_depth_m=useful_depth,
        misfit=misfit,
    )


# ---------------------------------------------------------------- parameters


def test_parameters_default_to_pacs_form() -> None:
    assert InversionParameters().model_dump() == {
        "n_layers": 2,
        "vs_layers": ({"vs_min": 100.0, "vs_max": 1_000.0, "vs_perturb_std": 20.0},) * 2,
        "thickness_layers": (
            {"thickness_min": 1.0, "thickness_max": 10.0, "thickness_perturb_std": 1.0},
        ),
        "n_iterations": 100_000,
        "n_burnin_iterations": 10_000,
        "n_chains": 5,
    }


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: InversionParameters(n_layers=3), r"vs_layers must have length n_layers \(3\)"),
        (
            lambda: InversionParameters(n_layers=3, vs_layers=(VsLayer(),) * 3),
            r"thickness_layers must have length n_layers - 1 \(2\)",
        ),
        (
            lambda: InversionParameters(n_iterations=2_000, n_burnin_iterations=1_900),
            r"n_iterations \(2000\) must exceed n_burnin_iterations \(1900\) by at least 150: "
            "each chain keeps one model every 150 iterations after the burn-in",
        ),
        (lambda: VsLayer(vs_min=500, vs_max=400), "vs_max must be greater than vs_min"),
        (
            lambda: ThicknessLayer(thickness_min=5, thickness_max=5),
            "thickness_max must be greater than thickness_min",
        ),
    ],
)
def test_parameters_are_checked(build: Callable[[], object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        build()


def test_the_burnin_follows_the_iterations() -> None:
    # A tenth, PAC's ratio: 2,000 iterations do not keep the default 10,000 of burn-in.
    assert InversionParameters(n_iterations=2_000).n_burnin_iterations == 200
    assert InversionParameters.model_validate({"n_iterations": 2e3}).n_burnin_iterations == 200
    # A burn-in the user gives is kept.
    assert InversionParameters(n_iterations=2_000, n_burnin_iterations=500).n_burnin_iterations == (
        500
    )


# ---------------------------------------------------------------- starting a job


def test_submit_records_a_queued_job_of_the_windows_g4_passed(
    picked: Picked, tmp_path: Path
) -> None:
    settings, folder = _copy(picked, tmp_path)

    record = submit_inversion(picked.run_id, SHORT, settings)

    assert re.fullmatch(r"inv-\d{8}-\d{6}-[0-9a-f]{4}", record.job_id)
    assert (record.run_id, record.state, record.total, record.windows, record.given) == (
        picked.run_id,
        "queued",
        3,
        (),
        SHORT,
    )
    assert read_record(folder) == record


def test_one_vs_range_is_submitted_for_every_layer(picked: Picked, tmp_path: Path) -> None:
    # The form invert's card offers, on a run G4 judged: checked as the job reads it, not
    # against PAC's default of 2 layers.
    settings, _ = _copy(picked, tmp_path)

    record = submit_inversion(
        picked.run_id, {"vs_layers": [{"vs_min": 100.0, "vs_max": 180.0}], **SHORT}, settings
    )

    assert record.state == "queued"
    assert record.given["vs_layers"] == [{"vs_min": 100.0, "vs_max": 180.0}] * 4


def test_an_inversion_needs_g4(picked: Picked, tmp_path: Path) -> None:
    settings, folder = _copy(picked, tmp_path)
    lines = (folder / "qc_log.jsonl").read_text().splitlines()
    kept = [line for line in lines if '"stage":"picking"' not in line]
    (folder / "qc_log.jsonl").write_text("\n".join(kept) + "\n")

    with pytest.raises(RunError, match=r"has not been judged up to G4: judge it"):
        submit_inversion(picked.run_id, SHORT, settings)


def test_values_that_cannot_hold_are_refused_at_once(picked: Picked, tmp_path: Path) -> None:
    settings, _ = _copy(picked, tmp_path)

    with pytest.raises(InversionError, match="must exceed n_burnin_iterations"):
        submit_inversion(
            picked.run_id, {"n_iterations": 1_000, "n_burnin_iterations": 900}, settings
        )
    with pytest.raises(InversionError, match="Extra inputs are not permitted"):
        submit_inversion(picked.run_id, {"iterations": 5}, settings)


@pytest.mark.parametrize("state", ["queued", "running"])
def test_one_inversion_per_run_at_a_time(picked: Picked, tmp_path: Path, state: str) -> None:
    settings, folder = _copy(picked, tmp_path)
    write_record(folder, _record(run_id=picked.run_id, state=state))

    with pytest.raises(InversionError, match=f"is already being inverted: .* is {state}"):
        submit_inversion(picked.run_id, SHORT, settings)


# ---------------------------------------------------------------- running a job


def test_every_window_g4_passed_is_inverted(inverted: Inverted) -> None:
    record = inverted.record

    assert (record.state, record.total, record.error) == ("succeeded", 3, None)
    assert record.started_at is not None and record.finished_at is not None
    assert [window.folder for window in record.windows] == list(PICKED)
    # The smooth median is reported at round depths down to half the longest wavelength.
    assert record.depths_m and record.depths_m[0] > 0
    # sigpipe's sampler sometimes fails a window twice, a chain keeping no predicted curve (more
    # often with this short sampler and 4 layers; none of 72 inversions at PAC's effort).
    failed = [window for window in record.windows if window.status == "failed"]
    assert len(failed) <= 1
    assert all(_sigpipes_known_failure(window.error) for window in failed)
    for window in record.windows:
        if window.status == "failed":
            continue
        # 4 layers asked, never fewer than 3: fewer when the curve resolves fewer.
        assert window.vs_m_s is not None and 3 <= len(window.vs_m_s) <= 4
        assert window.thicknesses_m is not None
        assert len(window.thicknesses_m) == len(window.vs_m_s) - 1
        assert window.vs_at_depths_m_s is not None
        assert len(window.vs_at_depths_m_s) == len(record.depths_m)
        assert window.useful_depth_m is not None and window.useful_depth_m > 0
        assert window.misfit is not None and window.misfit >= 0
    # The first pass reports its progress; the gates' retries follow.
    assert inverted.progress == [(done, 3) for done in range(4)]
    # What job_status reads is what the job returned.
    assert read_record(inverted.folder) == record


def test_g5s_retries_are_in_the_jobs_summary(inverted: Inverted) -> None:
    summary = inverted.record.summary

    assert summary is not None and summary.startswith("G5: ")
    # 3 models a chain: G5 asks, at once, the iterations 100 models a chain need. What comes of
    # it depends on the unseeded sampler: a window may need twice as many again, or a Vs bound
    # widened with them (its retry then named after that flag), and the line says "(each its
    # own)".
    (line,) = [line for line in summary.splitlines() if line.startswith("Retried G5:")][:1]
    assert '"n_iterations":17000,"n_burnin_iterations":1700' in line
    assert all(window.verdict in ("pass", "retry", "reject") for window in inverted.record.windows)
    # The settings the gates changed, from -> to, for the agent to report.
    first = inverted.record.changed[0]
    assert "n_iterations 500 -> 17000; n_burnin_iterations 50 -> 1700 at xmid " in first
    assert ", by G5:" in first


def test_the_section_of_the_models_g5_passed_is_saved_like_pacs(inverted: Inverted) -> None:
    # PAC's end-of-run outputs: the smooth median's section as a figure, every variant's in an
    # HDF5 file, over the models G5 passed (two at least; the sampler here is short).
    passed = [window for window in inverted.record.windows if window.verdict == "pass"]
    figure, grids = inverted.folder / SECTION_FIGURE, inverted.folder / SECTION_FILE
    if len(passed) < 2:
        assert not figure.exists()
        return
    assert figure.exists() and grids.exists()
    # And the picked curves against the ones the smooth medians predict, along the line.
    assert (inverted.folder / COMPARISON_FIGURE).exists()
    assert inverted.record.summary is not None
    assert f"Section of the {len(passed)} models G5 passed: {SECTION_FIGURE}." in (
        inverted.record.summary
    )
    with h5py.File(grids) as file:
        assert "smooth_median" in file


def test_a_section_needs_two_models(tmp_path: Path) -> None:
    assert save_section(tmp_path, ["xmid_1.00"]) is None
    assert save_comparison(tmp_path, ["xmid_1.00"]) is None
    assert not (tmp_path / SECTION_FIGURE).exists()


def _sigpipes_known_failure(error: str | None) -> bool:
    """sigpipe's inversion_mcmc when a chain kept no predicted curve."""
    return error is not None and ("dpred" in error or "could not be broadcast" in error)


def test_windows_get_pacs_files(inverted: Inverted) -> None:
    for window in inverted.record.windows:
        if window.status == "failed":
            assert _sigpipes_known_failure(window.error)
            continue
        files = {path.name for path in (inverted.folder / window.folder).iterdir()}
        assert files >= PAC_FILES


def test_a_window_whose_curve_is_gone_is_left_out(picked: Picked, tmp_path: Path) -> None:
    settings, folder = _copy(picked, tmp_path)
    (folder / "xmid_8.88" / "DispersionCurves_0000.csv").unlink()

    record = run_inversion_job(submit_inversion(picked.run_id, SHORT, settings), settings)

    assert record.state == "succeeded"
    assert [window.folder for window in record.windows] == ["xmid_14.88", "xmid_20.88"]


def test_150_iterations_after_the_burnin_are_enough(picked: Picked, tmp_path: Path) -> None:
    settings, _ = _copy(picked, tmp_path)
    # One model kept per chain (SAMPLE_EVERY is sigpipe's save_every).
    given = {"n_iterations": 300, "n_burnin_iterations": 150, "n_chains": 1}

    record = run_inversion_job(submit_inversion(picked.run_id, given, settings), settings)

    assert record.state == "succeeded"
    assert {window.status for window in record.windows} == {"succeeded"}


def test_a_job_that_cannot_run_is_failed(picked: Picked, tmp_path: Path) -> None:
    settings, folder = _copy(picked, tmp_path)
    record = submit_inversion(picked.run_id, SHORT, settings)
    # 149 iterations after the burn-in keep no model, which the checks before S4 refuse.
    record = record.model_copy(update={"given": {"n_iterations": 299, "n_burnin_iterations": 150}})

    record = run_inversion_job(record, settings)

    assert record.state == "failed"
    assert record.error is not None and "must exceed n_burnin_iterations" in record.error
    assert read_record(folder) == record


def test_find_job(inverted: Inverted) -> None:
    settings = Settings(input_dir=inverted.folder.parents[2], output_dir=inverted.folder.parents[1])

    assert find_job(inverted.record.job_id, settings) == (inverted.folder, inverted.record)
    for job_id in ("inv-20260923-000000-0000", "../x", "*"):
        with pytest.raises(InversionError, match=f"^Unknown job '{re.escape(job_id)}'"):
            find_job(job_id, settings)


# ---------------------------------------------------------------- the summary on its own


@pytest.mark.parametrize(("state", "live"), [("queued", False), ("running", False)])
def test_a_job_a_stopped_server_left_is_interrupted(state: str, live: bool) -> None:
    status = summarize_inversion(_record(state=state), live=live)

    assert status.state == "interrupted"
    assert status.error == "The server stopped before the job finished: call invert again."


def test_a_live_job_keeps_its_state() -> None:
    assert summarize_inversion(_record(state="running"), live=True).state == "running"


def test_summary_gives_the_range_of_the_models() -> None:
    record = _record(
        state="succeeded",
        finished_at=datetime(2026, 9, 23, 10, 2, 5, tzinfo=UTC),
        total=3,
        depths_m=(1.0, 2.0),
        windows=(
            _window(1.0, (200.0, 400.0), (3.0,), (200.0, 260.0), 4.0, 0.8),
            _window(2.0, (220.0, 500.0), (5.0,), (220.0, 240.0), 5.5, 1.26),
            WindowInversion(xmid=3.0, folder="xmid_3.00", status="failed", error="ValueError: x"),
        ),
    )

    status = summarize_inversion(record, live=False)

    assert status.model_dump(exclude={"job_id", "run_id"}) == {
        "state": "succeeded",
        "done": 3,
        "total": 3,
        "n_failed": 1,
        "elapsed_s": 125.0,
        "depths_m": (1.0, 2.0),
        "vs_m_s": ((200.0, 220.0), (240.0, 260.0)),
        "useful_depth_m": (4.0, 5.5),
        "misfit": (0.8, 1.26),
        "errors": ("xmid 3.00: ValueError: x",),
        "error": None,
        "summary": None,
        "changed": (),
    }


def test_the_summary_reports_nothing_of_models_before_any_window() -> None:
    status = summarize_inversion(_record(depths_m=(1.0, 2.0, 3.0)), live=True)

    assert (status.depths_m, status.vs_m_s, status.useful_depth_m, status.misfit) == (
        (1.0, 2.0, 3.0),
        (),
        None,
        None,
    )


def test_summary_reports_a_few_distinct_errors() -> None:
    def failed(xmid: float, error: str) -> WindowInversion:
        return WindowInversion(xmid=xmid, folder=f"xmid_{xmid:.2f}", status="failed", error=error)

    windows = [failed(1.0, "A"), failed(2.0, "A"), failed(3.0, "B"), failed(4.0, "C")]
    record = _record(windows=(*windows, failed(5.0, "D")), total=5)

    assert summarize_inversion(record, live=True).errors == (
        "xmid 1.00: A",
        "xmid 3.00: B",
        "xmid 4.00: C",
    )


# ---------------------------------------------------------------- the job manager


def test_jobs_run_one_at_a_time_and_stay_live_until_done() -> None:
    jobs = JobManager()
    release, second_started = threading.Event(), threading.Event()
    jobs.submit("first", lambda: release.wait(timeout=10))
    jobs.submit("second", second_started.set)

    # The second job waits for the first.
    assert not second_started.wait(timeout=0.2)
    assert jobs.is_live("first") and jobs.is_live("second")

    release.set()
    assert second_started.wait(timeout=10)
    # A job queued after both: once it runs, both are done.
    finished = threading.Event()
    jobs.submit("marker", finished.set)
    assert finished.wait(timeout=10)
    assert not jobs.is_live("first")
    assert not jobs.is_live("second")


def test_a_crashing_job_is_logged_and_forgotten(caplog: pytest.LogCaptureFixture) -> None:
    jobs = JobManager()
    finished = threading.Event()

    def crash() -> None:
        raise RuntimeError("boom")

    with caplog.at_level(logging.ERROR, logger="paco.jobs.manager"):
        jobs.submit("broken", crash)
        jobs.submit("marker", finished.set)
        assert finished.wait(timeout=10)

    assert not jobs.is_live("broken")
    assert "Job broken crashed" in caplog.text
