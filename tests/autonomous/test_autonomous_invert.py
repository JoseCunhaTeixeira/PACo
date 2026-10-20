"""invert, job_status, and the models after the freeze (docs/autonomous_plan.md §4.8, §10): on
active_test_profile's frozen curves (windows of 78 receivers on a coarse grid, 8 accepted and 11 rejected),
one pass then one group rerun, through the autonomous agent's tools at the quick effort; G5's and G6's
results in the journal and on the state card; each inversion seeded, the same seed making the same
models; each window's retries counted; the models decided as §8 says; look and advise on them."""

import shutil
from collections.abc import Iterator
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anyio
import pytest
from mcp import Client
from sigpipe.masw.inversion import InversionParameters
from sigpipe.masw.inversion.priors import MIN_LAYERS, Derived, derive_inversion
from sigpipe.masw.inversion.window import invert_window
from sigpipe.masw.picks import CURVES_FILE
from sigpipe.masw.runs import find_run

from paco.autonomous import server
from paco.autonomous.host.card import card
from paco.autonomous.tools import commit as committing
from paco.autonomous.tools import decide, freeze, given, invert, journal, pick
from paco.autonomous.tools.trials import Study, new_study
from paco.autonomous.tools.windows import select
from paco.call import shown
from paco.qc import inverting
from paco.qc.config import QCConfig
from paco.qc.inverting import QUICK_SAMPLER
from paco.qc.judging import saved_m0
from paco.qc.log import latest, read_attempts
from paco.qc.report import stretches, xmid_of
from paco.settings import Settings, get_settings

LINE = {"masw": {"length": 78}, "dispersion": {"nv": 250}}
REASONS = {"masw length": "the deepest windows whose curves pass", "dispersion nv": "a coarse grid"}
# On the line's pick, the 8 windows from 74.25 m on pass G3 and G4, accepted. The two beside them
# fail (on_data, narrow_span), and still do guided by their neighbours; the others find no ridge,
# or one standing out too little: all rejected.
ACCEPTED = [f"xmid_{74.25 + 1.5 * n:.2f}" for n in range(8)]
GUIDED = ["xmid_71.25", "xmid_72.75"]
REJECTED = (
    "no_ridge, on_data, narrow_span and prominence remain after the line's pick and a guide from "
    "the neighbours"
)
RERUN = {"windows": "xmids:[74.25, 75.75]", "parameters": {"n_iterations": 6000}}
# A model rejected after one quick pass, naming each fixable flag G5 may raise on it: the chains'
# seeds change with the run's id.
FIXABLE = (
    "not_converged, at_bound, thin_layer, alike_layers, too_deep or underfit after a quick pass, "
    "for a first look"
)


@dataclass(frozen=True)
class Inverted:
    """active_test_profile's frozen curves, inverted once, then a group again: what each call said."""

    settings: Settings
    study: Study
    run: Path
    said: dict[str, str]


@pytest.fixture(scope="module")
def inverted(input_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Inverted]:
    root = tmp_path_factory.mktemp("invert")
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(Settings.model_config, "env_file", None)
        patch.setenv("PACO_INPUT_DIR", str(input_dir))
        patch.setenv("PACO_OUTPUT_DIR", str(root / "output"))
        patch.setenv("PACO_WORKERS", "4")
        # As an evaluation sets it: every job quick, whatever the effort asked.
        patch.setenv("PACO_INVERSION_EFFORT", "quick")
        get_settings.cache_clear()
        settings = get_settings()
        config = QCConfig()
        study = new_study("active_test_profile", settings)
        line = committing.commit(study, LINE, REASONS, None, settings, config).line
        run = find_run(line.run_id, settings)
        base = committing.picking_of(study, line, settings, config)
        neighbours = pick.Recipe(how="guide", guide="neighbours")
        pick.pick(study, run, dict.fromkeys(GUIDED, neighbours), base, config, settings.workers)
        decide.decide(study, run, ACCEPTED, "accepted", "the near field noted", False, config)
        others = [path.name for path in sorted(run.glob("xmid_*")) if path.name not in ACCEPTED]
        decide.decide(study, run, others, "rejected", REJECTED, False, config)
        freeze.freeze(study, run, line)

        async def calls() -> dict[str, str]:
            said: dict[str, str] = {}
            async with Client(server.server) as client:

                async def call(name: str, arguments: dict[str, Any], meta: Any = None) -> str:  # noqa: ANN401
                    result = await client.call_tool(name, arguments, meta=meta)
                    return shown(result)

                for label, groups, effort in (
                    ("pass", [{"windows": "all"}], "quick"),
                    ("rerun", [RERUN], "full"),
                ):
                    queued = await call(
                        "invert",
                        {"profile": "active_test_profile", "groups": groups, "effort": effort},
                    )
                    job = queued.split()[1]
                    said[f"{label} queued"] = queued
                    said[label] = await call("job_status", {"job_id": job})
                said["too many"] = await call(
                    "invert",
                    {"profile": "active_test_profile", "groups": [RERUN], "effort": "quick"},
                    {"inversion_retries": 1},
                )
                said["accept"] = await call(
                    "decide",
                    {
                        "profile": "active_test_profile",
                        "windows": "xmids:[77.25]",
                        "status": "accepted",
                    },
                )
                said["reject untried"] = await call(
                    "decide",
                    {
                        "profile": "active_test_profile",
                        "windows": "xmids:[77.25]",
                        "status": "rejected",
                        "reason": "no use",
                    },
                )
                said["reject"] = await call(
                    "decide",
                    {
                        "profile": "active_test_profile",
                        "windows": "xmids:[77.25]",
                        "status": "rejected",
                        "reason": FIXABLE,
                    },
                )
                said["no model"] = await call(
                    "decide",
                    {
                        "profile": "active_test_profile",
                        "windows": "xmids:[69.75]",
                        "status": "rejected",
                        "reason": "no curve",
                    },
                )
                said["look"] = await call(
                    "look", {"profile": "active_test_profile", "what": "profile", "where": "74.25"}
                )
                said["advise"] = await call(
                    "advise", {"profile": "active_test_profile", "topic": "inversion"}
                )
                # The user gives the iterations: a group changing them is refused.
                given.give(study, given.Given(inversion={"n_iterations": 3000}))
                said["the user's"] = await call(
                    "invert",
                    {
                        "profile": "active_test_profile",
                        "groups": [
                            {"windows": "xmids:[78.75]", "parameters": {"n_iterations": 6000}}
                        ],
                        "effort": "quick",
                    },
                )
            return said

        yield Inverted(settings, study, run, anyio.run(calls))
    get_settings.cache_clear()


@pytest.mark.slow
def test_one_pass_then_one_group_rerun(inverted: Inverted) -> None:
    assert inverted.said["pass queued"].startswith("Job inv-")
    assert "8 windows, all {} at the quick effort" in inverted.said["pass queued"]
    assert inverted.said["pass"].splitlines()[0].startswith("Job inv-")
    assert ": succeeded, 8 of 8 windows" in inverted.said["pass"]
    assert ": succeeded, 2 of 2 windows" in inverted.said["rerun"]
    # The full effort asked, the quick one run: the server's setting, said.
    assert (
        "at the quick effort, PACO_INVERSION_EFFORT's (full was asked)"
        in (inverted.said["rerun queued"])
    )
    done = invert.inversions_of(inverted.study)
    assert [(one.unit, one.number) for one in done[8:]] == [("xmid_74.25", 2), ("xmid_75.75", 2)]
    assert {one.parameters["n_iterations"] for one in done[8:]} == {6000}
    assert {one.parameters["n_iterations"] for one in done[:8]} == {QUICK_SAMPLER["n_iterations"]}


@pytest.mark.slow
def test_job_status_groups_g5_by_symptom_then_g6(inverted: Inverted) -> None:
    lines = inverted.said["pass"].splitlines()

    assert lines[1] == "G5:"
    # One group a symptom (most often a single one, not_converged; the chains' seeds change with
    # the run's id), each with the spec that names its windows and no other, ready for invert
    # and decide.
    groups = [line for line in lines[2:] if line.startswith("- xmid ")]
    named: list[str] = []
    for group in groups:
        spec = group.rsplit("[windows: ", 1)[1].removesuffix("]")
        assert spec.startswith(("model:", "xmids:["))
        found = select(inverted.run, spec, {})
        assert group.startswith(f"- {stretches([xmid_of(unit) or 0.0 for unit in found])}: ")
        named += found
    # The accepted curves' models, the rejected windows' none.
    assert sorted(named) == ACCEPTED
    assert any(line.startswith("G6 on the line: ") for line in lines)


@pytest.mark.slow
def test_g5_and_g6_are_in_the_journal_and_on_the_card(inverted: Inverted) -> None:
    entries = [entry for entry in journal.entries(inverted.study) if entry.kind == "inversion"]

    assert [entry.id for entry in entries] == ["I1", "I2"]
    first, rerun = (entry.data for entry in entries)
    assert sorted(first["g5"]) == ACCEPTED
    assert sorted(rerun["g5"]) == ["xmid_74.25", "xmid_75.75"]
    assert first["g6"] in ("pass", "retry", "reject")
    assert entries[0].text.startswith("quick, 8 windows: G5 ")
    now = card(inverted.study, inverted.settings).splitlines()
    assert "Models (I2): 8 of 8 windows inverted" in now
    assert "  G5:" in now
    assert any(line.startswith("  G6 on the line: ") for line in now)
    assert "Retries: 6 a window at most; used: 1 at xmid 74.25-75.75 (2)" in now


@pytest.mark.slow
def test_each_inversion_is_seeded_and_logged(inverted: Inverted) -> None:
    attempts = read_attempts(inverted.run)
    for one in invert.inversions_of(inverted.study)[8:]:
        attempt = latest(attempts, one.unit, "inversion")
        assert attempt is not None
        assert one.seed == invert.seed_of(one.run_id, one.unit, one.number)
        assert f"seed {one.seed}" in attempt.notes
        assert attempt.triggered_by == "backtrack"


@pytest.mark.slow
def test_the_same_seed_makes_the_same_models(inverted: Inverted, tmp_path: Path) -> None:
    # sigpipe's Invert passes the seed on to inversion_mcmc, and its invert_window gives it.
    window = inverted.run / "xmid_74.25"
    curve = saved_m0(window / CURVES_FILE)
    assert curve is not None
    parameters = derive_inversion(curve, QCConfig().priors).parameters.model_copy(
        update={"n_iterations": 1200, "n_burnin_iterations": 300, "n_chains": 2}
    )
    models: list[str] = []
    for name, seed in (("a", 11), ("b", 11), ("c", 12)):
        shutil.copytree(
            window, tmp_path / name, ignore=shutil.ignore_patterns("SeismicInversion_*")
        )
        invert_window(tmp_path / name, parameters, chain_jobs=2, seed=seed)
        models.append((tmp_path / name / "SeismicInversion_Model_0000_median.csv").read_text())

    assert models[0] == models[1] != models[2]


@pytest.mark.slow
def test_a_window_past_its_retries_is_refused(inverted: Inverted) -> None:
    assert (
        "[bad argument] The windows at xmid 74.25-75.75 (2) have spent their 1 retries"
        in (inverted.said["too many"])
    )


@pytest.mark.slow
def test_the_models_are_decided_as_section_8_says(inverted: Inverted) -> None:
    accept, untried, reject = (inverted.said[key] for key in ("accept", "reject untried", "reject"))
    # What G5 said of 77.25's quick model: its chains' seeds change with the run's id.
    attempt = latest(read_attempts(inverted.run), "xmid_77.25", "inversion")
    assert attempt is not None
    verdict = attempt.results["G5"].verdict

    if verdict == "retry":
        assert "G5 did not pass its model" in accept
        assert "owes an inversion after its fixable" in untried
        assert reject.startswith("Rejected 1 window at xmid 77.25 (1): not_converged")
        assert reject.splitlines()[1].startswith("- 77.25 had: #1 3,000 iterations (retry)")
    elif verdict == "reject":  # a blocking flag: rejected without more
        assert "blocking flag" in accept
        assert untried.startswith("Rejected 1 window at xmid 77.25 (1): no use")
    else:  # the quick pass converged there: accepted, or a reason asked for its notes
        assert accept.startswith(("Accepted 1 window at xmid 77.25 (1)", "Nothing decided"))
        assert accept.startswith("Accepted") or "needs a reason" in accept


@pytest.mark.slow
def test_a_window_rejected_before_the_freeze_has_no_model_to_decide(inverted: Inverted) -> None:
    assert inverted.said["no model"] == (
        "Error executing tool decide: [bad argument] No frozen curve where xmids:[69.75]: no "
        "model to decide."
    )


@pytest.mark.slow
def test_a_group_changing_the_users_iterations_is_refused(inverted: Inverted) -> None:
    assert inverted.said["the user's"] == (
        "Error executing tool invert: [bad argument] n_iterations is the user's (3000): propose a "
        "change to them, never make it."
    )


@pytest.mark.slow
def test_look_and_advise_read_the_models(inverted: Inverted) -> None:
    assert inverted.said["look"].startswith(
        "xmid_74.25, its model's measures in its profile's stead (vision is off): fits: ensemble"
    )
    advice = inverted.said["advise"]
    assert advice.startswith("G5's changes of the inversion, by window (not applied):") or (
        advice == "G5 asks no window's inversion to change."
    )


@pytest.mark.slow
def test_g5_keeps_a_model_past_the_studys_longer_runs(inverted: Inverted, tmp_path: Path) -> None:
    # An inversion the agent asks again starts the run's log afresh: G5 weighs the study's runs
    # of the window, and keeps a model sampled longer twice already, with its warning.
    run = tmp_path / inverted.run.name
    shutil.copytree(inverted.run, run)
    unit = "xmid_74.25"
    curve = saved_m0(run / unit / CURVES_FILE)
    assert curve is not None
    derived = derive_inversion(curve, QCConfig().priors)
    short = replace(
        derived,
        parameters=derived.parameters.model_copy(
            update={"n_iterations": 1200, "n_burnin_iterations": 300, "n_chains": 2}
        ),
    )
    flags: dict[int, set[str]] = {}
    for longer in (0, 2):
        loop = inverting.Loop(MIN_LAYERS, longer, False)
        inverting.invert_windows(
            run,
            {unit: short},
            (5.0,),
            QCConfig(),
            "backtrack",
            2,
            seeds={unit: 11},
            loops={unit: loop},
        )
        attempt = latest(read_attempts(run), unit, "inversion")
        assert attempt is not None
        flags[longer] = {flag.name for flag in attempt.results["G5"].flags}

    assert "not_converged" in flags[0]
    assert "not_converged" not in flags[2]
    assert flags[2] & {"few_samples", "multimodal"}


def _derived(**values: int) -> Derived:
    return Derived(parameters=InversionParameters.model_validate(values), notes=(), reach_m=20.0)


def _sampler(derived: Derived) -> tuple[int, int, int]:
    one = derived.parameters
    return one.n_iterations, one.n_burnin_iterations, one.n_chains


def test_the_full_effort_runs_pacs_sampler_at_least() -> None:
    at_effort = invert._at_effort  # pyright: ignore[reportPrivateUsage]
    # A window looked at quickly, then inverted at the full effort: the quick sampler is not
    # carried over; longer runs stay, and so do the user's iterations.
    looked = _derived(n_iterations=3000, n_burnin_iterations=750, n_chains=2)
    assert _sampler(at_effort(looked, {}, {}, "full")) == (200_000, 50_000, 5)
    longer = _derived(n_iterations=400_000, n_burnin_iterations=100_000)
    assert _sampler(at_effort(longer, {}, {}, "full")) == (400_000, 100_000, 5)
    users = _derived(n_iterations=50_000, n_chains=2)
    assert _sampler(at_effort(users, {}, {"n_iterations": 50_000}, "full")) == (50_000, 12_500, 5)
    # Less asked at the full effort is refused, saying where it goes.
    check_full = invert._check_full  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(ValueError, match=r"not n_iterations 20,000\. A shorter pass is the quick"):
        check_full({"n_iterations": 20_000, "n_burnin_iterations": 5_000}, given.Given())
    check_full({"n_iterations": 400_000, "n_chains": 8}, given.Given())
    # The user's iterations, said again, stay theirs.
    check_full({"n_iterations": 2000}, given.Given(inversion={"n_iterations": 2000}))


def test_the_quick_effort_is_short_unless_given() -> None:
    at_effort = invert._at_effort  # pyright: ignore[reportPrivateUsage]
    # After a full run, the short sampler again; iterations given keep their quarter of burn-in.
    full = _derived(n_iterations=400_000)
    assert _sampler(at_effort(full, {}, {}, "quick")) == (3000, 750, 2)
    given = _derived(n_iterations=20_000)
    assert _sampler(at_effort(given, {"n_iterations": 20_000}, {}, "quick")) == (20_000, 5000, 2)
    users = _derived(n_iterations=50_000)
    assert _sampler(at_effort(users, {}, {"n_iterations": 50_000}, "quick")) == (50_000, 12_500, 2)


def test_g5s_loop_weighs_the_studys_runs_at_the_same_effort() -> None:
    loop = invert._loop  # pyright: ignore[reportPrivateUsage]

    def ran(effort: invert.Effort, iterations: int, *flags: str) -> invert.Inverted:
        return invert.Inverted(
            job_id="inv-1",
            run_id="run-1",
            unit="xmid_1.00",
            number=1,
            seed=1,
            effort=effort,
            parameters={"n_iterations": iterations, "n_layers": 4},
            flags=dict.fromkeys(flags, "fixable"),
            at=datetime.now(UTC),
        )

    own = [ran("quick", 3000, "not_converged"), ran("full", 200_000, "not_converged")]
    own.append(ran("full", 400_000, "not_converged"))
    # The quick look is not one of the full runs: 800,000 iterations sample longer a second time.
    assert loop(own, 800_000, "full") == inverting.Loop(MIN_LAYERS, 2, False)
    # Chains that disagreed, then as many iterations: the ranges were narrowed.
    assert loop(own[:2], 200_000, "full") == inverting.Loop(MIN_LAYERS, 0, True)
    # An underfit at 4 layers: no fewer than 5 after.
    assert loop([ran("full", 200_000, "underfit")], 200_000, "full").fewest_layers == 5
