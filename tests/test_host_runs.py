"""The host's part in settling the run a message works on (paco.agent.runs): the stages a plan
does on a run, and an offered call read back to make it."""

from paco.agent.runs import RunInfo, parse_call, to_do


def _run(went: str) -> RunInfo:
    label = f"active, windows of 24 receivers, {went}"
    return RunInfo("20261004-070000-aaaa", "active_p1", label, went)


def test_a_plan_does_what_its_run_lacks_for_the_stages_asked() -> None:
    # The models need curves: picked first on a run of images, not on one of curves.
    assert to_do(("invert",), _run("images")) == ("pick", "invert")
    assert to_do(("invert",), _run("curves")) == ("invert",)
    assert to_do(("soils",), _run("images")) == ("pick", "soils")
    # A new run: processed first, its curves picked for its models.
    assert to_do(("invert",), None) == ("process", "pick", "invert")
    assert to_do(("process",), None) == ("process",)


def test_an_offered_call_is_read_back_to_be_made() -> None:
    assert parse_call('pick(run_id="r", windows="all")') == (
        "pick",
        {"run_id": "r", "windows": "all"},
    )
    processing = (
        'run_processing(profile="active_p1", overrides={"masw": {"length": 24}}, again=true)'
    )
    assert parse_call(processing) == (
        "run_processing",
        {"profile": "active_p1", "overrides": {"masw": {"length": 24}}, "again": True},
    )
    # A call the user must fill, or one that does not read: not made.
    assert parse_call('pick(run_id="r", positions=["<m>"])') is None
    assert parse_call("pick(run_id=r)") is None
