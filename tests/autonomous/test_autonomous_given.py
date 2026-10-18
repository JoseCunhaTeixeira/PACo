"""The settings the user gave (docs/autonomous_plan.md §8, rule 1): read by the objective form, kept in the
study (the latest message's winning), applied by code to every trial, commit and inversion, a call
changing one refused; what they hold back said on the card and in the report. On active_test_profile's
24-receiver windows with a velocity grid the ground is faster than, as narrow_velocities types
it."""

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest
from autonomous_judged import JudgedLine
from pydantic import ValidationError
from sigpipe.masw.inversion.priors import derive_inversion
from sigpipe.masw.picks import CURVES_FILE
from sigpipe.masw.runs import find_run

from paco.autonomous.host import answer
from paco.autonomous.host.card import card
from paco.autonomous.host.scope import BLANK, GeoScope
from paco.autonomous.tools import given, journal
from paco.autonomous.tools.commit import Committed, commit
from paco.autonomous.tools.given import Given, asks, given_of
from paco.autonomous.tools.invert import _at_effort  # pyright: ignore[reportPrivateUsage]
from paco.autonomous.tools.trials import Study, new_study, try_settings
from paco.qc.config import QCConfig
from paco.qc.given import given_of as paco_given_of
from paco.qc.inverting import QUICK_SAMPLER
from paco.qc.judging import saved_m0
from paco.qc.log import append_attempt, latest, read_attempts
from paco.qc.models import Attempt, Flag, GateResult, Override, Reject
from paco.settings import Settings

# As narrow_velocities and tight_bounds type them.
TYPED = {
    "length_receivers": 24,
    "vmax_m_s": 350.0,
    "iterations": 2000,
    "vs_min_m_s": 150.0,
    "vs_max_m_s": 300.0,
}
USER = Given(
    line={"masw": {"length": 24}, "dispersion": {"vmax": 350.0}},
    inversion={"n_iterations": 2000, "vs_min": 150.0, "vs_max": 300.0},
)
LOCKED = "propose a change to them, never make it."


# ---------------------------------------------------------------- the form and the study


def test_the_form_reads_the_settings_typed() -> None:
    scope = GeoScope.model_validate(
        {**BLANK, "invert": True, "profile": "active_test_profile", **TYPED}
    )

    assert scope.given(1.5) == USER
    # A length in metres, in receivers on a line of 1.5 m.
    in_metres = GeoScope.model_validate({**BLANK, "length_m": 34.5, "fmin_hz": 5.0})
    assert in_metres.given(1.5) == Given(line={"masw": {"length": 24}, "dispersion": {"fmin": 5.0}})
    assert GeoScope.model_validate(BLANK).given(1.5) == Given()
    # Each range low then high: the form sent back with why.
    with pytest.raises(ValidationError, match=re.escape("vs_min_m_s (300) must be below")):
        GeoScope.model_validate({**BLANK, "vs_min_m_s": 300.0, "vs_max_m_s": 150.0})


def test_the_latest_message_giving_a_setting_wins(tmp_path: Path) -> None:
    study = Study("active_test_profile", tmp_path)

    assert given.give(study, USER) == USER.said()
    assert given.give(study, USER) == []
    assert given.give(study, Given(line={"dispersion": {"vmax": 600.0}})) == [
        "dispersion vmax 600 (was 350)"
    ]
    assert given_of(study).line == {"masw": {"length": 24}, "dispersion": {"vmax": 600.0}}
    assert given_of(study).inversion == USER.inversion


def test_a_study_begun_before_kept_its_objectives_length(tmp_path: Path) -> None:
    study = Study("active_test_profile", tmp_path)
    journal.objective(study, {"given": {"masw": {"length": 30}}}, "balanced")

    assert given_of(study) == Given(line={"masw": {"length": 30}})


def test_a_change_of_the_users_settings_is_refused() -> None:
    with pytest.raises(
        ValueError, match=re.escape(f"dispersion vmax is the user's (350): {LOCKED}")
    ):
        given.check_line({"dispersion": {"vmax": 500, "nv": 250}}, USER)
    given.check_line({"dispersion": {"vmax": 350, "nv": 250}, "masw": {"step": 1}}, USER)
    with pytest.raises(ValueError, match=re.escape(f"n_iterations is the user's (2000): {LOCKED}")):
        given.check_inversion({"n_iterations": 6000}, USER)
    with pytest.raises(
        ValueError, match=re.escape("vs_max is the user's (300 m/s, every layer's)")
    ):
        given.check_inversion({"free": {"vs_max": 600}}, USER)
    with pytest.raises(
        ValueError, match=re.escape("vs_min is the user's (150 m/s, every layer's)")
    ):
        given.check_inversion({"vs_layers": [{"vs_min": 150}, {"vs_min": 100}]}, USER)
    given.check_inversion({"n_iterations": 2000, "n_chains": 3, "free": {"max_layers": 4}}, USER)


def test_the_users_settings_go_into_every_call() -> None:
    # The line's: the model's changes without them, those the line does not hold yet.
    assert given.without(
        {"dispersion": {"vmax": 350, "nv": 250}, "masw": {"length": 24}}, USER
    ) == {"dispersion": {"nv": 250}}
    assert given.missing({"masw": {"length": 24}}, USER) == {"dispersion": {"vmax": 350.0}}
    assert given.with_line({"dispersion": {"nv": 250}}, USER) == {
        "dispersion": {"nv": 250, "vmax": 350.0},
        "masw": {"length": 24},
    }
    # The inversion's, locked: every layer's bounds under the free layering, one range for every
    # layer under the fixed one.
    rules = QCConfig().priors
    bounds = {"vs_min": 150.0, "vs_max": 300.0}
    assert given.with_inversion({"n_chains": 3}, USER, rules) == (
        {"n_chains": 3, "n_iterations": 2000, "free": bounds},
        {"n_iterations": 2000, "free": bounds},
    )
    assert given.with_inversion({"layering": "fixed", "n_layers": 4}, USER, rules) == (
        {"layering": "fixed", "n_layers": 4, "n_iterations": 2000, "vs_layers": [bounds]},
        {"n_iterations": 2000, "vs_layers": [bounds]},
    )


@pytest.mark.slow
def test_the_checks_keep_a_bound_the_user_gave(judged_line: JudgedLine) -> None:
    # 89.25's curve reaches 270 m/s: a top Vs of 200 m/s fails the checks before S4, kept.
    curve = saved_m0(judged_line.folder / "xmid_89.25" / CURVES_FILE)
    assert curve is not None
    user = Given(inversion={"n_iterations": 2000, "vs_max": 200.0})
    parameters, locked = given.with_inversion({}, user, QCConfig().priors)

    derived = derive_inversion(curve, QCConfig().priors, parameters, locked)

    assert derived.parameters.free.vs_max == 200.0
    (note,) = (one for one in derived.notes if one.startswith("free.vs_max 200 m/s"))
    assert "kept as given (the check sets" in note
    # A quick inversion keeps the user's iterations, the sampler's others short.
    quick = _at_effort(derived, parameters, "quick").parameters
    assert quick.n_iterations == 2000
    assert quick.n_chains == QUICK_SAMPLER["n_chains"]


def test_what_the_users_settings_hold_back(tmp_path: Path) -> None:
    now = datetime.now(UTC)

    def logged(unit: str, stage: str, result: GateResult | None, notes: tuple[str, ...]) -> None:
        append_attempt(
            tmp_path,
            Attempt(
                unit=unit,
                stage=stage,  # pyright: ignore[reportArgumentType]
                attempt=1,
                parameters={},
                triggered_by="initial",
                started_at=now,
                finished_at=now,
                status="succeeded",
                results={result.gate: result} if result is not None else {},
                notes=notes,
            ),
        )

    locked = Flag(
        name="locked",
        message="G2: it asks dispersion vmax 525 (given: 350), ...",
        stage="phase_shift",
        action=Reject(reason="locked, asks dispersion vmax 525 (given: 350)"),
        fixable=False,
    )
    longer = Flag(
        name="not_converged",
        message="The chains disagree.",
        stage="inversion",
        action=Override(stage="inversion", overrides={"n_iterations": 4000}),
    )
    for xmid in ("23.25", "24.75"):
        g2 = GateResult(gate="G2", unit=f"xmid_{xmid}", verdict="reject", flags=(locked,))
        logged(f"xmid_{xmid}", "phase_shift", g2, ())
    g5 = GateResult(gate="G5", unit="xmid_30.75", verdict="retry", flags=(longer,))
    kept = "free.vs_max 300 m/s below 1.1 times the curve's fastest velocity (330 m/s): kept as "
    logged("xmid_30.75", "inversion", g5, (f"{kept}given (the check sets 363 m/s).", "seed 7"))

    assert asks(tmp_path, USER) == [
        "xmid 23.25-24.75 (2): G2 rejected them, asking dispersion vmax 525 (given: 350)",
        "xmid 30.75 (1): G5 asks n_iterations 4000 (given: 2000)",
        "xmid 30.75 (1): the checks before S4 kept free.vs_max as given, asking 363 m/s",
    ]
    assert asks(tmp_path, Given()) == []


# ---------------------------------------------------------------- a study with them


@dataclass(frozen=True)
class Kept:
    """A study of active_test_profile whose user gave windows of 24 receivers and velocities up to 350
    m/s: what its calls said, and its first commit."""

    settings: Settings
    study: Study
    refused: dict[str, str]
    current: dict[str, object]
    first: Committed

    @property
    def run(self) -> Path:
        return find_run(self.first.line.run_id, self.settings)


@pytest.fixture(scope="module")
def kept(input_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Kept]:
    root = tmp_path_factory.mktemp("given")
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(Settings.model_config, "env_file", None)
        patch.chdir(root)
        settings = Settings(input_dir=input_dir, output_dir=root / "output", workers=4)
        config = QCConfig()
        study = new_study("active_test_profile", settings)
        given.give(study, Given(line=USER.line))
        refused = {
            "trial": _refused(
                lambda: try_settings(
                    study, [{"dispersion": {"vmax": 500}}], settings=settings, config=config
                )
            ),
            "commit": _refused(
                lambda: commit(
                    study,
                    {"dispersion": {"vmax": 800}},
                    {"dispersion vmax": "x"},
                    None,
                    settings,
                    config,
                )
            ),
        }
        trial = try_settings(
            study, [{"near_field_m": 1.0}], "spread:3", settings=settings, config=config
        )
        # The model's own change only: the user's are added, with their reason.
        changes, reasons = {"dispersion": {"nv": 250}}, {"dispersion nv": "coarse"}
        first = commit(study, changes, reasons, None, settings, config)
        yield Kept(settings, study, refused, trial.current, first)


@pytest.mark.slow
def test_a_trial_and_a_commit_changing_them_are_refused(kept: Kept) -> None:
    assert kept.refused == {
        "trial": f"dispersion vmax is the user's (350): {LOCKED}",
        "commit": f"dispersion vmax is the user's (350): {LOCKED}",
    }
    # Every trial runs with them.
    assert kept.current == {"masw": {"length": 24}, "dispersion": {"vmax": 350.0}}


@pytest.mark.slow
def test_the_first_commit_runs_with_them_and_logs_them(kept: Kept) -> None:
    (made,) = kept.first.line.commits

    assert made.changes == {"dispersion": {"nv": 250, "vmax": 350.0}, "masw": {"length": 24}}
    assert made.reasons == {
        "dispersion nv": "coarse",
        "dispersion vmax": given.REASON,
        "masw length": given.REASON,
    }
    # As PACo logs a user's settings: the run's own record of them.
    assert paco_given_of(kept.run) == {"processing": USER.line}
    notes = latest(read_attempts(kept.run), "line", "phase_shift")
    assert notes is not None
    assert "dispersion vmax: given by the user (L1)" in notes.notes


@pytest.mark.slow
def test_g2_rejects_the_windows_they_hold_back_and_the_card_says_so(kept: Kept) -> None:
    held = asks(kept.run, given_of(kept.study))

    # The ground is faster than 350 m/s under some windows: G2 asks a wider grid, which only the
    # user may give.
    assert len(held) == 1
    assert held[0].endswith(": G2 rejected them, asking dispersion vmax 525 (given: 350)")
    lines = card(kept.study, kept.settings).splitlines()
    assert lines[1] == "Kept as you gave: masw length 24; dispersion vmax 350"
    assert lines[-1] == (
        f"Held back by the user's settings (propose the change to them): {held[0]}"
    )
    assert answer.kept(kept.study) == ["masw length 24", "dispersion vmax 350"]
    assert answer.held_back(kept.study, kept.settings) == held


def _refused(call: Callable[[], object]) -> str:
    """What `call` was refused with."""
    with pytest.raises(ValueError) as error:
        call()
    return str(error.value)
