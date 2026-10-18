"""Trials (docs/autonomous_plan.md §6 try_settings, §13 M3): variants of the line's settings tried on its
windows and scored, each made once; their images given back by the cache when the line is imaged
with their settings; and the trial sets."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from sigpipe.masw.profiles import load_profile
from sigpipe.masw.windows import MASWWindow

from paco.autonomous.tools import trials
from paco.autonomous.tools.trials import (
    At,
    Experiment,
    Failing,
    Flagged,
    Holdout,
    LineState,
    Spread,
    Study,
    TrialRun,
    concrete,
    describe,
    line_windows,
    new_study,
    parse_windows,
    resolve,
    try_settings,
)
from paco.qc.config import QCConfig
from paco.settings import Settings

CURRENT = {"masw": {"length": 24}}
# A change of each cost: the windows' length (their images), the shots (the images of the windows
# whose shots it changes), the automatic pick (the picks alone).
VARIANTS: list[dict[str, Any]] = [
    {"masw": {"length": 32}},
    {"near_field_m": 1.0},
    {"picking": {"fmax": 30.0}},
]


@dataclass(frozen=True)
class Made:
    """A study of active_test_profile and its first experiment."""

    settings: Settings
    study: Study
    first: Experiment


@pytest.fixture(scope="module")
def made(input_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Made]:
    """A study of active_test_profile, and the current settings with VARIANTS tried on three windows spread
    along the line (a few seconds)."""
    root = tmp_path_factory.mktemp("trials")
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(Settings.model_config, "env_file", None)
        patch.chdir(root)
        settings = Settings(input_dir=input_dir, output_dir=root / "output", workers=2)
        study = new_study("active_test_profile", settings)
        first = try_settings(
            study, VARIANTS, "spread:3", current=CURRENT, settings=settings, config=QCConfig()
        )
        yield Made(settings, study, first)


@pytest.fixture(scope="module")
def windows(input_dir: Path) -> list[MASWWindow]:
    """active_test_profile's 73 windows of 24 receivers, at xmid 17.25 to 125.25 m, 1.5 m apart."""
    profile = load_profile("active_test_profile", Settings(input_dir=input_dir))
    return line_windows(profile, concrete(CURRENT, profile, QCConfig()))


@pytest.mark.slow
def test_the_current_settings_and_three_variants_are_scored(made: Made) -> None:
    first = made.first

    assert (first.id, first.repeated) == ("E1", False)
    assert [one.label for one in first.tried] == ["current", "1", "2", "3"]
    assert [one.changes for one in first.tried] == [{}, *VARIANTS]
    current = first.tried[0]
    # Three windows spread along the line, its ends included; the first two curves pass G3.
    assert first.xmids == current.scorecard.windows == (17.25, 71.25, 125.25)
    assert current.scorecard.covered == (17.25, 71.25)
    assert current.scorecard.depth_m is not None
    assert current.against is None
    assert all(one.against is not None for one in first.tried[1:])


@pytest.mark.slow
def test_each_change_remakes_what_it_changes(made: Made) -> None:
    runs = {one.label: _run(made.study, one.trial) for one in made.first.tried}

    # The records preprocessed once: no variant changes their preprocessing.
    assert len(list((made.study.folder / trials.RECORDS).iterdir())) == 1
    assert runs["current"].seconds["records"] > 0
    assert [runs[label].seconds["records"] for label in ("1", "2", "3")] == [0, 0, 0]
    # Windows of 32 receivers: other windows, their own images.
    assert (runs["1"].length, runs["1"].length_m) == (32, 46.5)
    assert [result.xmid for result in runs["1"].windows] == [23.25, 71.25, 119.25]
    # A near field of 1 m takes the shots 0.75 m off each window's ends (the shots are 6 m apart:
    # one at each end of these three, but 101.dat, which G1 rejected): every image made again.
    assert [result.cached for result in runs["2"].windows] == [False, False, False]
    # The pick alone: every image from the cache.
    assert all(result.cached for result in runs["3"].windows)
    assert runs["3"].seconds["images"] == 0


@pytest.mark.slow
def test_the_same_call_again_returns_its_experiment(made: Made) -> None:
    before = sorted((made.study.folder / trials.TRIALS).iterdir())

    again = try_settings(
        made.study,
        VARIANTS,
        "spread:3",
        current=CURRENT,
        settings=made.settings,
        config=QCConfig(),
    )

    assert (again.id, again.repeated) == (made.first.id, True)
    assert [one.trial for one in again.tried] == [one.trial for one in made.first.tried]
    assert all(one.taken and one.seconds == 0 for one in again.tried)
    assert sorted((made.study.folder / trials.TRIALS).iterdir()) == before


@pytest.mark.slow
def test_a_holdout_set_leaves_out_the_last_sets_windows(made: Made) -> None:
    holdout = try_settings(
        made.study,
        VARIANTS[2:],
        "holdout:3",
        current=CURRENT,
        settings=made.settings,
        config=QCConfig(),
    )

    assert holdout.id != made.first.id
    assert holdout.xmids == (18.75, 69.75, 123.75)


@pytest.mark.slow
def test_a_trials_images_come_back_when_the_line_is_imaged_with_its_settings(
    made: Made, tmp_path: Path
) -> None:
    # As a commit of variant 2's settings would: its records preprocessed again in a folder of
    # their own, judged by G1, and the line's windows imaged.
    config = QCConfig()
    profile = load_profile("active_test_profile", made.settings)
    trial = concrete({**CURRENT, "near_field_m": 1.0}, profile, config)
    records, preprocessed = trials.preprocessed(
        tmp_path, trial.preset, profile, made.settings, config
    )
    cone = trials.fast_cone(profile, trial.preset, config)
    exclusions, _, _ = trials.settled(tmp_path, profile, trial.preset, records, config, cone)
    line = line_windows(profile, trial)
    tried = {
        f"xmid_{result.xmid:.2f}" for result in _run(made.study, made.first.tried[2].trial).windows
    }
    chosen = [window for window in line if f"xmid_{window.xmid:.2f}" in tried] + [line[2]]

    outcomes = trials.image(chosen, trial.preset, records, tmp_path, exclusions, made.settings)

    assert preprocessed
    assert {outcome.folder: outcome.cached for outcome in outcomes} == {
        "xmid_17.25": True,
        "xmid_71.25": True,
        "xmid_125.25": True,
        "xmid_20.25": False,  # never imaged
    }


@pytest.mark.slow
def test_g1_is_made_once_for_the_trials_that_preprocess_alike(made: Made) -> None:
    # The variants change neither the records' preprocessing nor the images' band, and take the
    # same cone: one G1, made by the first trial and given to the others.
    (entry,) = (made.study.folder / trials.G1).iterdir()
    kept = trials.Settled.model_validate_json((entry / trials.SETTLED).read_text())

    assert [_run(made.study, one.trial).g1 for one in made.first.tried] == [kept.results] * 4
    assert kept.exclusions.records == ("101.dat", "129.dat")
    # Each record's spectra kept, for a run whose records are its own.
    assert len(list(entry.glob("records/*/Spectrum_0000.png"))) == 25


@pytest.mark.slow
def test_a_trial_changes_no_run(made: Made) -> None:
    profile = made.settings.output_dir / "active_test_profile"

    assert [path.name for path in profile.iterdir()] == [trials.STUDIES]


@pytest.mark.slow
def test_the_experiment_as_the_model_reads_it(made: Made) -> None:
    lines = describe(made.first).splitlines()

    assert lines[0] == "E1: 3 variants on spread:3, against the current settings:"
    assert lines[1].startswith("- current: 2 of 3 windows pass G3; depth ")
    # 32 receivers: the same two curves pass G3, shallower but surer: neither dominates. A pick up
    # to 30 Hz: as deep, less sure: the current settings dominate.
    assert lines[2].startswith(
        '- 1 {"masw": {"length": 32}}: 2 of 3 windows pass G3 (+0, -0); depth 8 m (P10 7.6 m), '
    )
    assert "dominated" not in lines[2]
    assert lines[4].startswith('- 3 {"picking": {"fmax": 30.0}}: 2 of 3 windows pass G3 (+0, -0); ')
    assert lines[4].endswith("; dominated by current")


@pytest.mark.slow
@pytest.mark.parametrize(
    ("variant", "said"),
    [
        ({"masw": {"step": 2}}, "one receiver apart"),
        ({"near_field_m": -1.0}, "near_field_m"),
        ({"picking": {"no_such": 1}}, "no_such"),
        ({"no_stage": {}}, "no_stage"),
    ],
)
def test_a_variant_that_cannot_be_tried_is_refused_before_anything_runs(
    made: Made, variant: dict[str, Any], said: str
) -> None:
    with pytest.raises(ValueError, match=said):
        try_settings(
            made.study, [variant], current=CURRENT, settings=made.settings, config=QCConfig()
        )


@pytest.mark.slow
def test_the_number_of_variants_is_bounded(made: Made) -> None:
    for variants in ([], [{"near_field_m": float(n)} for n in range(7)]):
        with pytest.raises(ValueError, match="variant"):
            try_settings(
                made.study, variants, current=CURRENT, settings=made.settings, config=QCConfig()
            )


def test_a_trial_set_in_words() -> None:
    assert parse_windows("failing+spread:9") == (Failing(), Spread(9))
    assert parse_windows("flag:near_field + xmids:[2.5, 3.25] + holdout:4") == (
        Flagged("near_field"),
        At((2.5, 3.25)),
        Holdout(4),
    )
    for spec in ("spread:0", "all", "xmids:[a]"):
        with pytest.raises(ValueError):
            parse_windows(spec)


@pytest.mark.slow
def test_the_windows_a_trial_set_names(windows: list[MASWWindow]) -> None:
    def named(
        spec: str, line: LineState | None = None, last: tuple[float, ...] = ()
    ) -> list[float]:
        return [
            window.xmid
            for window in resolve(parse_windows(spec), windows, line or LineState(), last)
        ]

    # Spread along the line, its ends included.
    assert named("spread:3") == [17.25, 71.25, 125.25]
    # The windows nearest the positions, each once.
    assert named("xmids:[0, 50.2, 50.4, 200]") == [17.25, 50.25, 125.25]
    # The committed line's windows failing, or carrying a flag; parts joined.
    line = LineState(
        verdicts={30.75: "retry", 60.75: "pass", 90.75: "reject"},
        flags={60.75: frozenset({"near_field"})},
    )
    assert named("failing", line) == [30.75, 90.75]
    assert named("failing+flag:near_field", line) == [30.75, 60.75, 90.75]
    # Spread over the windows the last set leaves.
    assert named("holdout:3", last=(17.25, 71.25, 125.25)) == [18.75, 69.75, 123.75]
    with pytest.raises(ValueError, match="commit one first"):
        named("failing")


def _run(study: Study, key: str) -> TrialRun:
    return TrialRun.model_validate_json(
        (study.folder / trials.TRIALS / key / trials.RESULT).read_text()
    )
