import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, tzinfo
from pathlib import Path
from typing import Any

import pytest

from paco.presets import PresetError, make_preset, resolve_preset
from paco.profiles import Profile, ProfileError, summarize
from paco.runs import (
    RunError,
    RunManifest,
    RunSummary,
    WindowOutcome,
    find_run,
    load_manifest,
    processing,
    run_processing,
    summarize_run,
)
from paco.settings import Settings
from paco.windows import MASWWindow, build_windows

# Four 24-receiver windows along the 96-receiver demo lines.
SMALL_WINDOWS = {"masw": {"length": 24, "step": 24}}
# A dispersion band too narrow to hold a frequency step: PACo leaves that rule to sigpipe, whose
# phase shift fails in every window.
FAILING = {**SMALL_WINDOWS, "dispersion": {"fmin": 10.1, "fmax": 10.2}}

WINDOW_FILES = {
    "active": {
        "window.json",
        "Stream_0000.png",
        "Stream_0001.png",
        "DispersionImage_0000.png",
        "DispersionImage_0000.hdf5",
    },
    "passive": {
        "window.json",
        "Stream_0000.png",
        "Stream_0000.hdf5",
        "DispersionImage_0000.png",
        "DispersionImage_0000.hdf5",
    },
}


@dataclass(frozen=True)
class Run:
    summary: RunSummary
    manifest: RunManifest
    folder: Path
    working_dir: Path  # where run_processing was called from
    progress: list[tuple[int, int]]


def _run(
    demo_input_dir: Path, root: Path, profile: str, preset: str, overrides: dict[str, Any]
) -> Run:
    settings = Settings(input_dir=demo_input_dir, output_dir=root / "outputs", workers=2)
    working_dir = root / "working_dir"
    working_dir.mkdir()
    progress: list[tuple[int, int]] = []

    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(working_dir)
        summary = run_processing(
            profile,
            preset,
            overrides,
            settings,
            on_progress=lambda done, total: progress.append((done, total)),
        )

    folder = settings.output_dir / summary.path
    manifest = RunManifest.model_validate_json((folder / "run.json").read_text())
    return Run(summary, manifest, folder, working_dir, progress)


# Real runs are the slow part: each one runs once and is shared by the tests below.


@pytest.fixture(scope="module")
def active_run(demo_input_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Run:
    root = tmp_path_factory.mktemp("active")
    return _run(demo_input_dir, root, "active_p1", "active", SMALL_WINDOWS)


@pytest.fixture(scope="module")
def passive_run(demo_input_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Run:
    root = tmp_path_factory.mktemp("passive")
    return _run(demo_input_dir, root, "passive_p1", "passive", SMALL_WINDOWS)


@pytest.fixture(scope="module")
def failing_run(demo_input_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Run:
    root = tmp_path_factory.mktemp("failing")
    return _run(demo_input_dir, root, "active_p1", "active", FAILING)


# ---------------------------------------------------------------- successful runs


def test_summary_counts_the_windows(active_run: Run) -> None:
    summary = active_run.summary

    assert re.fullmatch(r"\d{8}-\d{6}-[0-9a-f]{4}", summary.run_id)
    assert summary.model_dump(exclude={"run_id", "duration_s"}) == {
        "profile": "active_p1",
        "preset": "active",
        "path": f"active_p1/{summary.run_id}",
        "n_windows": 4,
        "n_processed": 4,
        "n_failed": 0,
        "n_skipped": 0,
        "errors": (),
    }


@pytest.mark.parametrize("kind", ["active", "passive"])
def test_run_folder_has_pacs_layout(request: pytest.FixtureRequest, kind: str) -> None:
    run: Run = request.getfixturevalue(f"{kind}_run")
    window_folders = {window.folder for window in run.manifest.windows}

    assert {path.name for path in run.folder.iterdir()} == {"run.json", "logs", *window_folders}
    for window in run.manifest.windows:
        assert window.folder == f"xmid_{window.xmid:.2f}"
        files = {path.name for path in (run.folder / window.folder).iterdir()}
        assert files == WINDOW_FILES[kind]


def test_sigpipe_logs_stay_inside_the_run(active_run: Run) -> None:
    assert (active_run.folder / "logs").is_dir()
    assert list(active_run.working_dir.iterdir()) == []


def test_manifest_records_what_the_run_used(active_run: Run, profiles: dict[str, Profile]) -> None:
    manifest = active_run.manifest
    profile = profiles["active_p1"]

    assert manifest.run_id == active_run.summary.run_id
    assert manifest.profile == summarize(profile)
    assert manifest.preset == resolve_preset(make_preset("active", SMALL_WINDOWS), profile)
    assert set(manifest.versions) == {"paco", "sigpipe"}
    assert manifest.started_at <= manifest.finished_at
    assert manifest.n_positions == 4

    xmids = [window.xmid for window in manifest.windows]
    assert xmids == sorted(xmids)
    for window in manifest.windows:
        assert window.status == "succeeded"
        assert window.error is None
        assert window.duration_s is not None and window.duration_s > 0


def test_window_json_is_the_window_processed(active_run: Run, profiles: dict[str, Profile]) -> None:
    expected = build_windows(profiles["active_p1"], active_run.manifest.preset.masw)
    saved = [
        MASWWindow.model_validate_json(
            (active_run.folder / window.folder / "window.json").read_text()
        )
        for window in active_run.manifest.windows
    ]

    assert saved == expected


def test_progress_goes_from_zero_to_every_window(active_run: Run) -> None:
    assert active_run.progress == [(done, 4) for done in range(5)]


@pytest.mark.parametrize("kind", ["active", "passive", "failing"])
def test_summary_stays_short(request: pytest.FixtureRequest, kind: str) -> None:
    # The summary is what the agent reads: it must stay far below the tool-output budget.
    run: Run = request.getfixturevalue(f"{kind}_run")

    assert len(run.summary.model_dump_json()) < 1_000


# ---------------------------------------------------------------- failing windows


def test_failing_windows_do_not_stop_the_run(failing_run: Run) -> None:
    summary = failing_run.summary
    first_xmid = failing_run.manifest.windows[0].xmid

    assert (summary.n_windows, summary.n_processed, summary.n_failed) == (4, 0, 4)
    # Every window fails for the same reason, which is reported once.
    (error,) = summary.errors
    assert error.startswith(
        f"xmid {first_xmid:.2f}: ValueError: no frequencies found in the requested band"
    )


def test_failed_windows_keep_their_traceback(failing_run: Run) -> None:
    for window in failing_run.manifest.windows:
        assert window.status == "failed"
        assert window.duration_s is None
        assert window.error is not None
        assert window.error.startswith("ValueError: no frequencies found")

        log = (failing_run.folder / window.folder / "error.log").read_text()
        # The worker's own traceback, down to the sigpipe step that failed.
        assert "_RemoteTraceback" in log
        assert "phase_shift.py" in log


# ---------------------------------------------------------------- requests refused before any work


def test_no_valid_shot_is_refused_before_writing(demo_settings: Settings) -> None:
    overrides = {"masw": {"distance_min": 50, "distance_max": 60}}

    with pytest.raises(RunError, match=r"all 94 positions were skipped\. Widen masw\.distance_min"):
        run_processing("active_p1", "active", overrides, demo_settings)
    assert not demo_settings.output_dir.exists()


@pytest.mark.parametrize(
    ("profile", "preset", "overrides", "error", "message"),
    [
        ("nope", "active", {}, ProfileError, "Unknown profile 'nope'"),
        ("active_p1", "activ", {}, PresetError, "Unknown preset 'activ'"),
        ("passive_p1", "active", {}, PresetError, "only fits active profiles"),
        ("active_p1", "active", {"masw": {"lenght": 24}}, PresetError, r"masw\.lenght: unknown"),
        (
            "active_p1",
            "active",
            {"masw": {"length": 97}},
            ValueError,
            r"length \(97\) exceeds the 96 receivers",
        ),
    ],
)
def test_invalid_requests_are_refused_before_writing(
    demo_settings: Settings,
    profile: str,
    preset: str,
    overrides: dict[str, Any],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        run_processing(profile, preset, overrides, demo_settings)
    assert not demo_settings.output_dir.exists()


# ---------------------------------------------------------------- run IDs


class _FrozenClock:
    """Stands in for datetime in paco.runs.processing: every call is the same second."""

    @staticmethod
    def now(tz: tzinfo | None = None) -> datetime:
        return datetime(2026, 9, 23, 10, 0, 0, tzinfo=tz)


def test_run_ids_stay_unique_within_one_second(
    demo_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The second run draws the first run's suffix again, then a new one.
    suffixes = iter(["abcd", "abcd", "ef01"])

    def token_hex(_nbytes: int) -> str:
        return next(suffixes)

    monkeypatch.setattr(processing, "datetime", _FrozenClock)
    monkeypatch.setattr(processing.secrets, "token_hex", token_hex)
    # A single 3-receiver window, with a dispersion band that makes it fail: cheap to run.
    overrides = {"masw": {"length": 3, "step": 94}, "dispersion": {"fmin": 10.1, "fmax": 10.2}}

    first = run_processing("active_p1", "active", overrides, demo_settings)
    second = run_processing("active_p1", "active", overrides, demo_settings)

    assert (first.run_id, second.run_id) == ("20260923-100000-abcd", "20260923-100000-ef01")


# ---------------------------------------------------------------- the summary on its own


def test_summary_reports_a_few_distinct_errors(profiles: dict[str, Profile]) -> None:
    def failed(xmid: float, error: str) -> WindowOutcome:
        return WindowOutcome(xmid=xmid, folder=f"xmid_{xmid:.2f}", status="failed", error=error)

    long_error = "ValueError: " + "x" * 300
    started_at = datetime(2026, 9, 23, 10, 0, 0, tzinfo=UTC)
    manifest = RunManifest(
        run_id="20260923-100000-abcd",
        profile=summarize(profiles["active_p1"]),
        preset=make_preset("active"),
        versions={"paco": "1.0.0", "sigpipe": "1.0.0"},
        started_at=started_at,
        finished_at=started_at + timedelta(seconds=12.34),
        n_positions=10,
        windows=(
            WindowOutcome(xmid=0.5, folder="xmid_0.50", status="succeeded", duration_s=1.0),
            failed(1.0, "ValueError: a"),
            failed(2.0, "ValueError: a"),
            failed(3.0, "ValueError: b"),
            failed(4.0, long_error),
            failed(5.0, "TypeError: c"),
        ),
    )

    summary = summarize_run(manifest)

    assert (summary.n_windows, summary.n_processed, summary.n_failed) == (6, 1, 5)
    assert summary.n_skipped == 4
    assert summary.duration_s == 12.3
    # Distinct messages only, at most three, each cut to 200 characters.
    assert summary.errors == (
        "xmid 1.00: ValueError: a",
        "xmid 3.00: ValueError: b",
        f"xmid 4.00: {long_error[:200]}",
    )


# ---------------------------------------------------------------- finding runs


def _fake_run(settings: Settings, profile: str, run_id: str) -> Path:
    """A run folder with a run.json: all find_run looks for."""
    folder = settings.output_dir / profile / run_id
    folder.mkdir(parents=True)
    (folder / "run.json").write_text("{}")
    return folder


def test_find_run_returns_the_run_folder(settings: Settings) -> None:
    folder = _fake_run(settings, "passive_p1", "20260923-100000-abcd")

    assert find_run("20260923-100000-abcd", settings) == folder


def test_an_unknown_run_lists_the_five_latest(settings: Settings) -> None:
    for second in range(7):
        profile = ("active_p1", "passive_p1")[second % 2]
        _fake_run(settings, profile, f"20260923-10000{second}-abcd")
    # An interrupted run: its folder has no run.json.
    (settings.output_dir / "active_p1" / "20260923-100009-abcd").mkdir()

    with pytest.raises(RunError) as error:
        find_run("20260923-100009-abcd", settings)

    assert str(error.value) == (
        "Unknown run '20260923-100009-abcd'. Latest runs: 20260923-100006-abcd (active_p1), "
        "20260923-100005-abcd (passive_p1), 20260923-100004-abcd (active_p1), "
        "20260923-100003-abcd (passive_p1), 20260923-100002-abcd (active_p1)."
    )


def test_an_unknown_run_without_any_run(settings: Settings) -> None:
    with pytest.raises(
        RunError, match=r"^Unknown run '20260923-100000-abcd'\. Latest runs: none\.$"
    ):
        find_run("20260923-100000-abcd", settings)


@pytest.mark.parametrize(
    "run_id", ["*", "2026*", "../passive_p1/20260923-100000-abcd", "20260923-100000-abcd/.."]
)
def test_only_run_ids_are_looked_up(settings: Settings, run_id: str) -> None:
    # A run exists, but a pattern or a path must not reach it, nor anything outside the outputs.
    _fake_run(settings, "passive_p1", "20260923-100000-abcd")

    with pytest.raises(RunError, match=r"^Unknown run"):
        find_run(run_id, settings)


def test_load_manifest_reads_run_json(active_run: Run, demo_input_dir: Path) -> None:
    settings = Settings(input_dir=demo_input_dir, output_dir=active_run.folder.parents[1])

    assert load_manifest(active_run.summary.run_id, settings) == active_run.manifest
