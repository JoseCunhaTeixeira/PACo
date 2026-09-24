"""The evaluation suite of the QC loop (docs/qc_workflow.md, milestone 14): looking around,
processing and picking with the gates' own fixes, the loop's changes to the settings the user
typed (said in the answer), the whole line to Vs models with no setting at all, and the cases
where the agent is stuck and must ask. Every scenario where the data decide checks that the
agent asked nothing, and that the thresholds stayed the configuration's."""

from dataclasses import dataclass

from paco.evaluation.checks import (
    Check,
    answer_mentions,
    any_of,
    asked_nothing,
    asked_the_user,
    at_most_calls,
    called,
    curves,
    excluded,
    in_order,
    inversion_succeeded,
    loop_retried,
    models,
    never_called,
    no_inversion_started,
    no_settings_invented,
    not_succeeded,
    only_called,
    retried_value,
    succeeded,
    thresholds_unchanged,
)
from paco.evaluation.models import Kind

SMALL_WINDOWS = {"masw": {"length": 24, "step": 24}}


@dataclass(frozen=True)
class Scenario:
    name: str
    kind: Kind
    questions: tuple[str, ...]  # what the user types, one message after another
    checks: tuple[Check, ...]
    rubric: str  # what the judge model grades


SCENARIOS = (
    Scenario(
        name="list_profiles",
        kind="look around",
        questions=("Which seismic profiles can I process?",),
        checks=(
            called("list_profiles"),
            answer_mentions("active_p1", "passive_p1"),
            at_most_calls(3),
        ),
        rubric="The answer lists the profiles PACo offers, and nothing else.",
    ),
    Scenario(
        name="describe_profile",
        kind="look around",
        questions=(
            "How many receivers does active_p1 have, and how far apart are they, in metres?",
        ),
        checks=(
            called("inspect_profile", profile="active_p1"),
            answer_mentions("96", "0.25"),
            at_most_calls(3),
        ),
        rubric="The answer gives 96 receivers, 0.25 m apart, read from the profile.",
    ),
    Scenario(
        name="unknown_profile",
        kind="recover",
        questions=("Describe the profile active_p2.",),
        checks=(
            called("inspect_profile"),
            answer_mentions("active_p1", "passive_p1"),
            at_most_calls(4),
        ),
        rubric="active_p2 does not exist: the agent says so, names the profiles that do exist, "
        "and does not describe a profile it made up.",
    ),
    Scenario(
        name="pick_active",
        kind="the loop",
        questions=(
            "Process active_p1 with windows of 24 receivers, every 24 receivers, and pick the "
            "curves.",
        ),
        checks=(
            only_called("run_processing", profile="active_p1", overrides=SMALL_WINDOWS),
            in_order("run_processing", "pick"),
            # The demo's records start 20 ms before the shot: G1 corrects it. The pick of xmid
            # 14.88 jumps between modes: G3 picks it again.
            loop_retried("G1:shifted_trigger"),
            loop_retried("G3:mode_jump"),
            answer_mentions(curves),
            never_called("invert"),
            asked_nothing(),
            thresholds_unchanged(),
        ),
        rubric="The agent processes with the requested windows and picks the curves, then says "
        "how many curves passed and what the gates fixed or rejected (the trigger delay, the "
        "window whose pick jumped), without asking anything.",
    ),
    Scenario(
        name="dead_trace",
        kind="the loop",
        questions=(
            "Process active_dead with windows of 24 receivers, every 24 receivers, and pick the "
            "curves.",
        ),
        checks=(
            succeeded("pick"),
            excluded("1.mseed", 40),
            any_of(answer_mentions("40"), answer_mentions("dead")),
            asked_nothing(),
            thresholds_unchanged(),
        ),
        rubric="Trace 40 of the first record is dead: the agent reports that G1 left it out, "
        "and how many curves passed, without asking anything.",
    ),
    Scenario(
        name="narrow_velocities",
        kind="the loop",
        questions=(
            "Process active_p1 with windows of 24 receivers, every 24 receivers, and velocities "
            "up to 250 m/s, and pick the curves.",
        ),
        checks=(
            succeeded("pick"),
            # The ground is faster than 250 m/s in places: G2 widens the range, and the agent
            # must say that a setting the user typed changed.
            loop_retried("G2:ridge_at_vmax"),
            answer_mentions(retried_value("phase_shift", "dispersion", "vmax")),
            asked_nothing(),
            thresholds_unchanged(),
        ),
        rubric="The velocity range the user typed is too narrow for the ground: the agent says "
        "the gates widened it (to what) and any other setting that changed, and how many curves "
        "passed, without asking anything.",
    ),
    Scenario(
        name="few_iterations",
        kind="the loop",
        questions=(
            "Process active_p1 with windows of 24 receivers, every 24 receivers, pick the curves "
            "and invert them quickly, with 2,000 iterations.",
        ),
        checks=(
            called("invert", parameters={"n_iterations": 2000}),
            # 12 models a chain: G5 asks the iterations that give enough.
            loop_retried("G5:not_converged"),
            inversion_succeeded(),
            answer_mentions(retried_value("inversion", "n_iterations")),
            asked_nothing(),
            thresholds_unchanged(),
        ),
        rubric="2,000 iterations are too few: the agent starts the inversion as asked, follows "
        "the job to its end, and says that G5 raised the iterations (to what) and what the "
        "models are, without asking anything.",
    ),
    Scenario(
        name="tight_bounds",
        kind="the loop",
        questions=(
            "Process active_p1 with windows of 24 receivers, every 24 receivers, pick the curves "
            "and invert them with shear-wave velocities between 100 and 180 m/s, 20,000 "
            "iterations.",
        ),
        checks=(
            called("invert"),
            inversion_succeeded(),
            # The curves reach 290 m/s: the checks before S4 widen the upper bound.
            answer_mentions("180"),
            asked_nothing(),
            thresholds_unchanged(),
        ),
        rubric="The bounds the user typed do not bracket the curves: the agent says the checks "
        "before the inversion widened them (to what), and what the models are, without asking "
        "anything.",
    ),
    Scenario(
        name="zero_settings",
        kind="the loop",
        questions=("Process active_p1 and give me the Vs models.",),
        checks=(
            no_settings_invented("run_processing"),
            in_order("run_processing", "pick", "invert"),
            inversion_succeeded(),
            answer_mentions(models),
            asked_nothing(),
            thresholds_unchanged(),
        ),
        rubric="With no setting at all, the agent runs the whole loop (processing, picking, the "
        "inversion, followed to its end) and gives the Vs models and why some windows have "
        "none, without asking anything.",
    ),
    Scenario(
        name="no_curve",
        kind="stuck",
        questions=(
            "Process passive_p1 with windows of 24 receivers, every 24 receivers, pick the "
            "curves and invert them.",
        ),
        checks=(
            succeeded("run_processing"),
            not_succeeded("invert"),
            no_inversion_started(),
            asked_the_user(),
            thresholds_unchanged(),
        ),
        rubric="No curve of the passive line passes the gates, so there is nothing to invert: "
        "the agent says so with the main reasons, and asks the user which to try, with a few "
        "concrete options.",
    ),
    Scenario(
        name="windows_too_long",
        kind="stuck",
        questions=("Process active_p1 with windows of 120 receivers.",),
        checks=(
            answer_mentions("96"),
            asked_the_user(),
            never_called("invert"),
        ),
        rubric="The line has only 96 receivers: the agent explains the limit and asks the user "
        "which length to use, with a few concrete options. It never reports a run with windows "
        "of 120 receivers.",
    ),
)
