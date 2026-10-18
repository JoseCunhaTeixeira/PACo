"""The A/B scenarios (docs/geo_plan.md §12): the same messages for PACo's agent and the geophysicist
agent, on active_p2 (closer to real data than PAC's demo line, active_p1), its defect copies and
settings typed wrong.

PACo's agent is scored by PACo's own checks, imported: its scenario as it is where the message is
its suite's (zero_settings), else its suite's checks that still apply. The geophysicist agent is
scored by its own (paco.geo.evaluation.checks) and PACo's that read what both leave, the runs:
where PACo's refuse a change (the ladder's length kept, no setting invented), its own expect
experiments, commits from them and lengths compared; where a setting the user typed blocks the
objective, PACo's expect the window left out over it, its own a question proposing the change
(§11).

Every message asks for the Vs models, since the geophysicist agent always goes on to them, and
gives no window step, since its windows are one receiver apart (§8, rule 3): with a length given,
both agents then work on the same windows. PACo's scenarios, on active_p2 too, are reworded so: the
same line, and the values typed wrong for its ground (its curves run from about 240 to 790 m/s)."""

from dataclasses import dataclass, replace

from paco.evaluation.checks import (
    Check,
    answer_mentions,
    any_of,
    asked_nothing,
    asked_the_user,
    called,
    checks_ask,
    excluded,
    inversion_succeeded,
    kept_as_given,
    ladder_length_kept,
    line_muted,
    locked_asks,
    refused_as_locked,
    thresholds_unchanged,
    top_vs_kept,
)
from paco.evaluation.scenarios import SCENARIOS, Scenario
from paco.geo.evaluation.checks import (
    committed_from_experiments,
    curves_frozen,
    every_run_with,
    experimented,
    lengths_tried,
    models_decided,
    objective_read,
    report_given,
)
from paco.geo.evaluation.known_truth import TRUTHS, Truth


@dataclass(frozen=True)
class Pair:
    """One scenario both agents play: PACo's agent's (its messages are both agents'), and the
    geophysicist agent's checks and rubric. `profile`: the line it plays on; `truth`, a
    known-truth scenario's model, whose curves the line's are."""

    name: str
    paco: Scenario
    checks: tuple[Check, ...]
    rubric: str
    profile: str = "active_p2"
    truth: Truth | None = None

    @property
    def questions(self) -> tuple[str, ...]:
        return self.paco.questions


def _paco(name: str) -> Scenario:
    """PACo's scenario `name`, as its suite has it."""
    return next(scenario for scenario in SCENARIOS if scenario.name == name)


def _reworded(name: str, question: str, *checks: Check, rubric: str) -> Scenario:
    """PACo's scenario `name` with `question` in place of its message, its checks that still apply
    and the rubric saying so."""
    return replace(_paco(name), questions=(question,), checks=checks, rubric=rubric)


# Every play of the geophysicist agent that runs to its report: the whole way, by experiments.
WHOLE_WAY = (
    report_given(),
    curves_frozen(),
    models_decided(),
    experimented(),
    committed_from_experiments(),
    thresholds_unchanged(),
    asked_nothing(),
)
# The first record of active_p2's defect copies, as obspy writes it again.
FIRST_RECORD = "101.mseed"

PAIRS = (
    Pair(
        name="zero_settings",
        # PACo's zero_settings, on active_p2: its checks name no line.
        paco=replace(
            _paco("zero_settings"), questions=("Process active_p2 and give me the Vs models.",)
        ),
        checks=WHOLE_WAY,
        rubric="With no setting at all, the agent surveys the line (96 receivers 1.5 m apart, 97 "
        "shots), chooses its settings by experiments, picks and decides every window, freezes "
        "the curves, inverts them and decides every model; the report gives the Vs models, the "
        "depth they inform and the windows left out with why, without asking anything.",
    ),
    Pair(
        name="dead_trace",
        profile="active_dead",
        paco=_reworded(
            "dead_trace",
            "Process active_dead with windows of 24 receivers and give me the Vs models.",
            excluded(FIRST_RECORD, 40),
            any_of(answer_mentions("40"), answer_mentions("dead")),
            inversion_succeeded(),
            asked_nothing(),
            thresholds_unchanged(),
            rubric="Trace 40 of the first record is dead: the agent reports that G1 left it out, "
            "how many curves passed, and the Vs models, without asking anything.",
        ),
        checks=(
            excluded(FIRST_RECORD, 40),
            every_run_with("masw", "length", value=24),
            any_of(answer_mentions("40"), answer_mentions("dead")),
            *WHOLE_WAY,
        ),
        rubric="Trace 40 of the first record is dead: G1 leaves it out by itself; the windows of "
        "24 receivers the user gave stay; the agent chooses the other settings by experiments "
        "and reports the dead trace, the windows left out and the Vs models, without asking "
        "anything.",
    ),
    Pair(
        name="refractions",
        profile="active_refracted",
        paco=_reworded(
            "refractions",
            "Process active_refracted and give me the Vs models.",
            line_muted(),
            answer_mentions("mute"),
            inversion_succeeded(),
            asked_nothing(),
            thresholds_unchanged(),
            rubric="Refractions swamp the surface waves: the agent says the line was muted (the "
            "mute trial's choice, and why), how many curves passed, and the Vs models, without "
            "asking anything.",
        ),
        checks=(line_muted(), answer_mentions("mute"), *WHOLE_WAY),
        rubric="Refractions swamp the surface waves: the survey shows the fast arrivals, an "
        "experiment shows a mute bringing the surface waves back, the line is committed muted "
        "with that reason, and the report says so with the Vs models, without asking anything.",
    ),
    Pair(
        name="narrow_velocities",
        paco=_reworded(
            "narrow_velocities",
            "Process active_p2 with windows of 24 receivers and velocities up to 350 m/s, and "
            "give me the Vs models.",
            kept_as_given("phase_shift", "dispersion", "vmax", value=350),
            refused_as_locked("G2"),
            answer_mentions("350", locked_asks("G2")),
            thresholds_unchanged(),
            rubric="The velocity range the user typed is too narrow for the ground: it stays as "
            "given; the agent says which windows G2 left out over it and the range G2 asks, and "
            "suggests processing again with that range.",
        ),
        checks=(
            every_run_with("dispersion", "vmax", value=350),
            every_run_with("masw", "length", value=24),
            asked_the_user(),
            answer_mentions("350"),
            thresholds_unchanged(),
        ),
        rubric="The ground is faster than the 350 m/s the user typed: the images' ridges reach "
        "the edge of the velocity grid. The agent keeps 350 m/s in every commit and asks the "
        "user to widen it, saying which windows it cuts and the range it would use.",
    ),
    Pair(
        name="few_iterations",
        paco=_reworded(
            "few_iterations",
            "Process active_p2 with windows of 24 receivers, pick the curves and invert them "
            "quickly, with 2,000 iterations.",
            called("invert", parameters={"n_iterations": 2000}),
            kept_as_given("inversion", "n_iterations", value=2000),
            refused_as_locked("G5"),
            answer_mentions(locked_asks("G5")),
            thresholds_unchanged(),
            rubric="2,000 iterations are too few: the agent inverts as asked, says the "
            "iterations stayed as given, which windows G5 left out over them and the iterations "
            "it asks, and suggests inverting again with those.",
        ),
        checks=(
            kept_as_given("inversion", "n_iterations", value=2000),
            every_run_with("masw", "length", value=24),
            curves_frozen(),
            asked_the_user(),
            answer_mentions("2000"),
            thresholds_unchanged(),
        ),
        rubric="2,000 iterations are too few for chains to converge: the agent inverts the frozen "
        "curves with them, reads G5's convergence, and asks the user for more iterations, "
        "saying how many and which windows need them; it never changes them itself.",
    ),
    Pair(
        name="tight_bounds",
        paco=_reworded(
            "tight_bounds",
            "Process active_p2 with windows of 24 receivers, pick the curves and invert them with "
            "shear-wave velocities between 150 and 300 m/s, 20,000 iterations.",
            called("invert"),
            inversion_succeeded(),
            top_vs_kept(300),
            answer_mentions("300", checks_ask),
            thresholds_unchanged(),
            rubric="The bounds the user typed do not bracket the curves: they stay as given; the "
            "agent says the checks before the inversion ask a wider upper bound (which), what "
            "the models are, and suggests inverting again with it.",
        ),
        checks=(
            top_vs_kept(300),
            every_run_with("masw", "length", value=24),
            curves_frozen(),
            asked_the_user(),
            answer_mentions("300"),
            thresholds_unchanged(),
        ),
        rubric="The curves reach 500 m/s and more, above the 300 m/s the user typed as Vs's upper "
        "bound: the agent inverts with the user's bounds, reads the samples piled at the bound, "
        "and asks the user to widen it, saying to what; it never changes it itself.",
    ),
    Pair(
        name="deeper",
        paco=_reworded(
            "deeper",
            "Process active_p2 and give me the Vs models, reaching as deep as this line allows.",
            ladder_length_kept(),
            inversion_succeeded(),
            asked_nothing(),
            thresholds_unchanged(),
            rubric="The agent keeps the window length run_processing chose, the best curves, and "
            "says down to which depth the curves and the models reach, without asking anything.",
        ),
        checks=(objective_read("depth"), lengths_tried(2), *WHOLE_WAY),
        rubric="Depth first: the agent compares window lengths and shot distances for the depth "
        "they reach and the windows they keep, commits the deepest that keeps the line covered, "
        "and reports down to which depth the models are informed, without asking anything.",
    ),
    Pair(
        name="detail",
        paco=_reworded(
            "detail",
            "Process active_p2 and give me the Vs models, with as much lateral detail as the data "
            "allow.",
            ladder_length_kept(),
            inversion_succeeded(),
            asked_nothing(),
            thresholds_unchanged(),
            rubric="The agent keeps the window length run_processing chose, the best curves, and "
            "says how many windows the line has, how many curves passed and the Vs models, "
            "without asking anything.",
        ),
        checks=(objective_read("lateral"), lengths_tried(2), *WHOLE_WAY),
        rubric="Lateral detail first: the agent compares short window lengths for the windows "
        "they keep and their precision, commits the shortest that keeps reliable curves, and "
        "reports the windows, the gaps and the Vs models, without asking anything.",
    ),
)


# The known-truth set (paco.geo.evaluation.known_truth): PACo's curves_invert on active_p2 (a run
# whose curves are picked, "Invert active_p2."), its models at PAC's effort, which the score
# judges; the geophysicist agent from its P4 on the same curves.
TRUTH_PAIRS = tuple(
    Pair(
        name=f"truth_{truth.name}",
        paco=replace(
            _paco("curves_invert"),
            name=f"truth_{truth.name}",
            questions=("Invert active_p2.",),
            full_inversion=True,
        ),
        checks=(report_given(), models_decided(), thresholds_unchanged(), asked_nothing()),
        rubric="The curves are frozen: the agent inverts them, groups the windows by G5's "
        "symptoms and changes one thing a group within the retries, decides every model, and "
        "reports the models and the depth they inform, without asking anything.",
        truth=truth,
    )
    for truth in TRUTHS
)


def pairs_named(names: list[str], lines: frozenset[str]) -> tuple[Pair, ...]:
    """The scenarios `names` names (every one when none, the known-truth set's too), of those
    whose line `lines` holds. Raises ValueError naming a scenario that does not exist."""
    every = (*PAIRS, *TRUTH_PAIRS)
    known = {pair.name for pair in every}
    if unknown := sorted(set(names) - known):
        raise ValueError(
            f"Unknown scenario(s): {', '.join(unknown)}. Scenarios: {', '.join(sorted(known))}."
        )
    return tuple(
        pair for pair in every if (not names or pair.name in names) and pair.profile in lines
    )
