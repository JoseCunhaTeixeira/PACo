"""invert, job_status, and the models after the freeze (docs/autonomous_plan.md §4.8, §10, §13 M7): on
active_test_profile's frozen curves (windows of 90 receivers on a coarse grid, 6 accepted and one rejected),
one pass then one group rerun, through the autonomous agent's tools at the quick effort; G5's and G6's
results in the journal and on the state card; each inversion seeded, the same seed making the same
models; each window's retries counted; the models decided as §8 says; look and advise on them."""

import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anyio
import pytest
from mcp import Client
from sigpipe.masw.inversion.priors import derive_inversion
from sigpipe.masw.picks import CURVES_FILE
from sigpipe.masw.runs import find_run

from paco.autonomous import server
from paco.autonomous.host.card import card
from paco.autonomous.tools import commit as committing
from paco.autonomous.tools import decide, freeze, given, invert, journal, pick
from paco.autonomous.tools.trials import Study, new_study
from paco.autonomous.tools.windows import select
from paco.autonomous.upstream.inversion import invert_window
from paco.call import shown
from paco.qc.config import QCConfig
from paco.qc.inverting import QUICK_SAMPLER
from paco.qc.judging import saved_m0
from paco.qc.log import latest, read_attempts
from paco.qc.report import stretches, xmid_of
from paco.settings import Settings, get_settings

LINE = {"masw": {"length": 90}, "dispersion": {"nv": 250}}
REASONS = {"masw length": "the deepest windows", "dispersion nv": "a coarse grid"}
# Three windows fail G3 on the line's pick (no_ridge, narrow_span): guided by their neighbours,
# two pass, the third keeps its narrow span and is rejected.
GUIDED = ["xmid_66.75", "xmid_68.25", "xmid_69.75"]
REJECTED = "xmid_69.75"
RERUN = {"windows": "xmids:[72.75, 74.25]", "parameters": {"n_iterations": 6000}}
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
        pick.pick(study, run, GUIDED, neighbours, base, config, settings.workers)
        decide.decide(study, run, _accepted(run), "accepted", "the near field noted", False, config)
        decide.decide(
            study,
            run,
            [REJECTED],
            "rejected",
            "narrow_span remains after a guide from the neighbours",
            False,
            config,
        )
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
                        "windows": "xmids:[66.75]",
                        "status": "accepted",
                    },
                )
                said["reject untried"] = await call(
                    "decide",
                    {
                        "profile": "active_test_profile",
                        "windows": "xmids:[66.75]",
                        "status": "rejected",
                        "reason": "no use",
                    },
                )
                said["reject"] = await call(
                    "decide",
                    {
                        "profile": "active_test_profile",
                        "windows": "xmids:[66.75]",
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
                    "look", {"profile": "active_test_profile", "what": "profile", "where": "72.75"}
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
                            {"windows": "xmids:[71.25]", "parameters": {"n_iterations": 6000}}
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
    assert "6 windows, all {} at the quick effort" in inverted.said["pass queued"]
    assert inverted.said["pass"].splitlines()[0].startswith("Job inv-")
    assert ": succeeded, 6 of 6 windows" in inverted.said["pass"]
    assert ": succeeded, 2 of 2 windows" in inverted.said["rerun"]
    # The full effort asked, the quick one run: the server's setting, said.
    assert (
        "at the quick effort, PACO_INVERSION_EFFORT's (full was asked)"
        in (inverted.said["rerun queued"])
    )
    done = invert.inversions_of(inverted.study)
    assert [(one.unit, one.number) for one in done[6:]] == [("xmid_72.75", 2), ("xmid_74.25", 2)]
    assert {one.parameters["n_iterations"] for one in done[6:]} == {6000}
    assert {one.parameters["n_iterations"] for one in done[:6]} == {QUICK_SAMPLER["n_iterations"]}


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
    # The accepted curves' models, the rejected window's none.
    assert sorted(named) == _accepted(inverted.run)
    assert any(line.startswith("G6 on the line: ") for line in lines)


@pytest.mark.slow
def test_g5_and_g6_are_in_the_journal_and_on_the_card(inverted: Inverted) -> None:
    entries = [entry for entry in journal.entries(inverted.study) if entry.kind == "inversion"]

    assert [entry.id for entry in entries] == ["I1", "I2"]
    first, rerun = (entry.data for entry in entries)
    assert sorted(first["g5"]) == _accepted(inverted.run)
    assert sorted(rerun["g5"]) == ["xmid_72.75", "xmid_74.25"]
    assert first["g6"] in ("pass", "retry", "reject")
    assert entries[0].text.startswith("quick, 6 windows: G5 ")
    now = card(inverted.study, inverted.settings).splitlines()
    assert "Models (I2): 6 of 6 windows inverted" in now
    assert "  G5:" in now
    assert any(line.startswith("  G6 on the line: ") for line in now)
    assert "Retries: 6 a window at most; used: 1 at xmid 72.75-74.25 (2)" in now


@pytest.mark.slow
def test_each_inversion_is_seeded_and_logged(inverted: Inverted) -> None:
    attempts = read_attempts(inverted.run)
    for one in invert.inversions_of(inverted.study)[6:]:
        attempt = latest(attempts, one.unit, "inversion")
        assert attempt is not None
        assert one.seed == invert.seed_of(one.run_id, one.unit, one.number)
        assert f"seed {one.seed}" in attempt.notes
        assert attempt.triggered_by == "backtrack"


@pytest.mark.slow
def test_the_same_seed_makes_the_same_models(inverted: Inverted, tmp_path: Path) -> None:
    # sigpipe's Invert passes the seed on to inversion_mcmc: the copy of invert_window gives it.
    window = inverted.run / "xmid_72.75"
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
        "[bad argument] The windows at xmid 72.75-74.25 (2) have spent their 1 retries"
        in (inverted.said["too many"])
    )


@pytest.mark.slow
def test_the_models_are_decided_as_section_8_says(inverted: Inverted) -> None:
    accept, untried, reject = (inverted.said[key] for key in ("accept", "reject untried", "reject"))
    # What G5 said of 66.75's quick model: its chains' seeds change with the run's id.
    attempt = latest(read_attempts(inverted.run), "xmid_66.75", "inversion")
    assert attempt is not None
    verdict = attempt.results["G5"].verdict

    if verdict == "retry":
        assert "G5 did not pass its model" in accept
        assert "owes an inversion after its fixable" in untried
        assert reject.startswith("Rejected 1 window at xmid 66.75 (1): not_converged")
        assert reject.splitlines()[1].startswith("- 66.75 had: #1 3,000 iterations (retry)")
    elif verdict == "reject":  # a blocking flag: rejected without more
        assert "blocking flag" in accept
        assert untried.startswith("Rejected 1 window at xmid 66.75 (1): no use")
    else:  # the quick pass converged there: accepted, or a reason asked for its notes
        assert accept.startswith(("Accepted 1 window at xmid 66.75 (1)", "Nothing decided"))
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
        "xmid_72.75, its model's measures in its profile's stead (vision is off): fits: ensemble"
    )
    advice = inverted.said["advise"]
    assert advice.startswith("G5's changes of the inversion, by window (not applied):") or (
        advice == "G5 asks no window's inversion to change."
    )


def _accepted(run: Path) -> list[str]:
    """The line's windows but the rejected one."""
    return [path.name for path in sorted(run.glob("xmid_*")) if path.name != REJECTED]
