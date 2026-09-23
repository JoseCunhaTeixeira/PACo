"""The first evaluation suite: two scenarios for each kind of task the agent must handle."""

from dataclasses import dataclass
from typing import Literal

from paco.evaluation.checks import (
    Check,
    answer_mentions,
    asked_the_user,
    at_most_calls,
    called,
    good_windows,
    in_order,
    inversion_succeeded,
    job_id,
    never_called,
    no_inversion_started,
    not_succeeded,
    only_called,
    succeeded,
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
    # How the simulated user answers an approval question.
    approval: Literal["accept", "decline"] = "decline"


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
        rubric="The answer lists the two profiles, active_p1 and passive_p1, and nothing else.",
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
        name="judge_active",
        kind="process and judge",
        questions=(
            "Process active_p1 with windows of 24 receivers, every 24 receivers, then tell me "
            "how many windows are good.",
        ),
        checks=(
            only_called("run_processing", profile="active_p1", overrides=SMALL_WINDOWS),
            in_order("run_processing", "dispersion_quality"),
            answer_mentions(good_windows),
            at_most_calls(6),
            never_called("invert"),
        ),
        rubric="The agent processes with the requested windows, judges the run, and reports the "
        "number of good windows that dispersion_quality returned.",
    ),
    Scenario(
        name="judge_passive",
        kind="process and judge",
        questions=(
            "Process passive_p1 with windows of 24 receivers, every 24 receivers, judge the "
            "windows, and pick the good ones.",
        ),
        checks=(
            only_called("run_processing", profile="passive_p1", overrides=SMALL_WINDOWS),
            succeeded("dispersion_quality"),
            not_succeeded("pick"),
            never_called("invert"),
        ),
        rubric="No window is good, so there is nothing to pick: the agent says so plainly, "
        "gives the main reasons (flags) and the advice dispersion_quality returned, and does not "
        "claim any curve was picked.",
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
        name="windows_too_long",
        kind="recover",
        questions=("Process active_p1 with windows of 120 receivers.",),
        # Qwen3-4B went on to judge, pick, and ask to invert.
        checks=(called("run_processing"), answer_mentions("96"), never_called("invert")),
        rubric="The line has only 96 receivers: the agent explains the limit and asks which "
        "length to use, or uses a shorter one and says so. It never reports a run with windows "
        "of 120 receivers.",
    ),
    Scenario(
        name="inversion_declined",
        kind="approval",
        questions=(
            "Process active_p1 with windows of 24 receivers, every 24 receivers, judge and pick "
            "the curves, then invert them.",
        ),
        checks=(
            only_called("run_processing", profile="active_p1", overrides=SMALL_WINDOWS),
            called("invert"),
            asked_the_user(),
            no_inversion_started(),
        ),
        rubric="The user declined the inversion when asked: the agent reports that no inversion "
        "started, does not claim otherwise, and asks what to change.",
        approval="decline",
    ),
    Scenario(
        name="inversion_approved",
        kind="approval",
        questions=(
            "Process active_p1 with windows of 24 receivers, every 24 receivers, judge and pick "
            "the curves, then invert them quickly, with 2,000 iterations and a single chain.",
        ),
        checks=(
            # Qwen3-4B once processed with the default length, then chose 48 receivers on the
            # quality advice, and inverted those windows without a word.
            only_called("run_processing", profile="active_p1", overrides=SMALL_WINDOWS),
            called("invert", parameters={"n_iterations": 2000, "n_chains": 1}),
            asked_the_user(),
            succeeded("invert"),
            # A job can start and still fail in every window: Qwen3-4B's 2,000 iterations
            # kept PAC's 10,000 of burn-in, and left nothing to sample.
            inversion_succeeded(),
            answer_mentions(job_id),
        ),
        rubric="The user approved: the agent starts the inversion with the requested effort, "
        "gives the job ID, and says how to follow it; it does not invent results the job has "
        "not reported.",
        approval="accept",
    ),
)
