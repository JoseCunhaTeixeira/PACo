"""The scope of a message: the form the model fills, checked in code, the calls within it, and
what the user and the model read of it."""

import json
from typing import Any

import anyio
import pytest
from openai.types.chat import ChatCompletionFunctionToolParam, ChatCompletionMessageParam
from pydantic import ValidationError

from paco import prompts
from paco.agent.model import Filled, Reply
from paco.agent.scope import (
    SCHEMA,
    Context,
    Offer,
    Scope,
    ScopeError,
    checked,
    continued,
    offers_in,
    read_scope,
    refusal,
)

NOTHING: dict[str, Any] = {
    "process": False,
    "pick": False,
    "invert": False,
    "soils": False,
    "profile": None,
    "run_id": None,
    "positions_m": [],
    "length_receivers": None,
    "length_m": None,
    "step_receivers": None,
    "step_m": None,
    "compare_lengths_receivers": [],
    "compare_lengths_m": [],
    "redo": False,
    "replace_hand_work": False,
    "option": None,
    "workers": None,
    "mode": None,
}


def _scope(**fields: Any) -> Scope:  # noqa: ANN401
    return Scope.model_validate({**NOTHING, **fields})


class FormModel:
    """Stands in for the model filling forms: its fills in order, and what it was sent."""

    def __init__(self, *fills: str) -> None:
        self._fills = list(fills)
        self.sent: list[list[ChatCompletionMessageParam]] = []

    async def __call__(
        self,
        messages: list[ChatCompletionMessageParam],  # noqa: ARG002
        tools: list[ChatCompletionFunctionToolParam],  # noqa: ARG002
    ) -> Reply:
        raise AssertionError("the scope is a form, not a conversation")

    async def fill(
        self,
        messages: list[ChatCompletionMessageParam],
        schema: dict[str, Any],
    ) -> Filled:
        assert schema == SCHEMA
        self.sent.append(list(messages))
        return Filled(content=self._fills.pop(0))


def test_the_form_makes_the_model_write_every_field_and_nothing_else() -> None:
    assert set(SCHEMA["required"]) == set(Scope.model_fields)
    assert SCHEMA["additionalProperties"] is False
    assert SCHEMA["properties"]["positions_m"]["maxItems"] == 10


def test_the_model_fills_the_form_with_the_message_and_what_the_host_knows() -> None:
    model = FormModel(json.dumps({**NOTHING, "pick": True, "invert": True, "option": 2}))
    offers = (
        Offer("complete the windows without a curve", 'pick(run_id="r", windows="missing")'),
        Offer("pick every window again", 'pick(run_id="r", windows="all")'),
    )
    context = Context(offers=offers, profile="active_p1", run_id="20260930-161253-89f5")

    read = anyio.run(read_scope, model, "The second one, then invert.", context)

    assert read.scope.asked == {"pick", "invert"} and read.scope.option == 2
    ((system, *examples, user),) = model.sent
    assert system == {"role": "system", "content": prompts.prompt("scope")}
    # The worked examples, as earlier messages and their forms.
    assert len(examples) == 2 * len(prompts.scope_examples())
    first = json.loads(str(examples[1].get("content")))
    assert first == {**NOTHING, **prompts.scope_examples()[0][1]}
    assert user.get("content") == (
        'Offered last: (1) complete the windows without a curve: pick(run_id="r", '
        'windows="missing"); (2) pick every window again: pick(run_id="r", windows="all").\n'
        "Current profile: active_p1; current run: 20260930-161253-89f5.\n"
        "Message: The second one, then invert."
    )


def test_a_form_that_does_not_parse_goes_back_once_with_its_error() -> None:
    fixed = FormModel('{"pick": true}', json.dumps({**NOTHING, "pick": True}))

    read = anyio.run(read_scope, fixed, "Pick active_p1.", Context())

    assert read.scope.pick and len(read.fills) == 2
    said_back = fixed.sent[1][-1].get("content")
    assert isinstance(said_back, str) and said_back.startswith("Not valid: process: Field required")
    # The second form's error, when neither parses.
    with pytest.raises(ScopeError, match="Invalid JSON"):
        anyio.run(read_scope, FormModel("{}", "not json"), "Pick active_p1.", Context())


def test_code_takes_out_what_it_can_tell_is_not_so() -> None:
    scope = _scope(run_id="active_p1", option=3, positions_m=[9.0, -2.0])

    kept = checked(scope, Context(offers=(Offer("a", "a()"),)))

    assert (kept.run_id, kept.option, kept.positions_m) == (None, None, [9.0])
    run = "20260930-161253-89f5"
    assert checked(_scope(run_id=run), Context()).run_id == run


def test_calls_outside_the_scope_are_refused_to_the_model() -> None:
    pick = _scope(pick=True)
    invert = _scope(invert=True)
    look = _scope()

    # Reading is always within; an earlier stage serves a later one, as the run needs it.
    assert refusal(look, "inspect", '{"what": "runs"}') is None
    assert refusal(invert, "run_processing", '{"profile": "active_p1"}') is None
    assert refusal(invert, "pick", '{"run_id": "r"}') is None
    assert refusal(pick, "redo", '{"run_id": "r", "stage": "phase_shift"}') is None
    assert refusal(pick, "invert", '{"run_id": "r"}') == (
        "Not called: this message asks pick, and invert is outside it. Do what it asks, or ask "
        "the user whether they want more."
    )
    assert refusal(pick, "redo", '{"run_id": "r", "stage": "inversion"}') is not None
    assert refusal(invert, "invert_petro", '{"run_id": "r", "model": "m"}') is not None
    assert refusal(look, "pick", '{"run_id": "r"}') == (
        "Not called: this message asks to look at what exists, running nothing, and pick is "
        "outside it. Do what it asks, or ask the user whether they want more."
    )
    # Asked only to process: processing, which gives the choice of a new run or one there.
    assert refusal(_scope(process=True), "pick", '{"run_id": "r"}') == (
        "Not called: this message asks process, and pick is outside it. Call run_processing: "
        "when the profile has runs, it gives the user's choice of a new run or the run to work on."
    )


def test_a_comparison_of_lengths_processes_no_line() -> None:
    compared = _scope(process=True, profile="active_p1", compare_lengths_m=[3.0, 6.0])
    then_picked = _scope(process=True, pick=True, compare_lengths_m=[3.0, 6.0])

    assert refusal(compared, "compare", '{"profile": "active_p1"}') is None
    assert refusal(compared, "run_processing", '{"profile": "active_p1"}') == (
        "Not called: this message compares window lengths, and run_processing is outside it: "
        "compare needs no run. Compare them; processing the line with one is the user's to ask."
    )
    # Curves asked as well: the line is processed for them.
    assert refusal(then_picked, "run_processing", '{"profile": "active_p1"}') is None


def test_the_user_and_the_model_read_the_scope() -> None:
    scope = _scope(
        pick=True, invert=True, profile="active_p1", positions_m=[9.0], redo=True, option=1
    )
    offer = Offer("pick every window again", 'pick(run_id="r", windows="all")')

    assert scope.line() == ("Scope: pick, invert · active_p1 · at 9 m · again · option 1.")
    assert scope.for_model(offer) == (
        "[PACo] This message asks pick, invert; profile active_p1; positions 9 m; to do again; "
        "not to replace the work made by hand. The user chose (1) pick every window again: "
        'pick(run_id="r", windows="all"). Make that call, then the rest this message asks.'
    )
    assert _scope().line() == "Scope: look only."
    assert scope.for_server() == {
        "asked": ["pick", "invert"],
        "redo": True,
        "hand_work": "unsaid",
        "positions_m": [9.0],
        "window": {},
        "compared": {},
        "workers": None,
        "mode": None,
    }


def test_the_options_a_result_offers_are_kept() -> None:
    result = json.dumps({"run_id": "r", "options": [{"label": "keep it", "call": "pick()"}]})

    assert offers_in(result) == (Offer("keep it", "pick()"),)
    assert offers_in('{"run_id": "r"}') == () == offers_in("Error executing tool pick")


def test_the_windows_the_message_gives_are_read_in_their_unit() -> None:
    receivers = _scope(process=True, profile="active_p1", length_receivers=24, step_receivers=24)
    metres = _scope(process=True, profile="active_p1", length_m=6.0)

    # The server sets them, converting metres itself; the user checks them on the scope line.
    assert receivers.for_server()["window"] == {"length": 24, "step": 24}
    assert metres.for_server()["window"] == {"length_m": 6.0}
    assert receivers.line() == (
        "Scope: process · active_p1 · windows of 24 receivers, every 24 receivers."
    )
    assert "windows of 6 m" in metres.for_model(None)
    assert _scope().for_server()["window"] == {}


def test_the_lengths_a_comparison_names_are_read_in_their_unit() -> None:
    metres = _scope(process=True, profile="active_p1", compare_lengths_m=[3.0, 6.0])
    receivers = _scope(process=True, compare_lengths_receivers=[12, 24, 48])

    assert metres.for_server()["compared"] == {"length_m": [3.0, 6.0]}
    assert receivers.for_server()["compared"] == {"length": [12.0, 24.0, 48.0]}
    assert metres.line() == "Scope: process · active_p1 · comparing windows of 3, 6 m."
    assert "comparing windows of 12, 24, 48 receivers" in receivers.for_model(None)


def test_the_workers_the_message_asks_are_read_and_given_every_stage() -> None:
    scope = _scope(process=True, profile="active_p1", workers=10)

    assert scope.line() == "Scope: process · active_p1 · 10 workers."
    assert "10 workers, which PACo gives every stage" in scope.for_model(None)
    assert scope.for_server()["workers"] == 10
    with pytest.raises(ValidationError):
        _scope(workers=0)


def test_a_choice_goes_on_with_what_the_message_that_got_the_options_asked() -> None:
    request = _scope(process=True, invert=True, profile="active_p2", positions_m=[30.0], workers=10)
    again = Offer(
        "process again, a new run (this one stays)",
        'run_processing(profile="active_p2", again=true)',
    )

    went_on = continued(_scope(process=True, redo=True, option=5), request, again)

    # The option says how (a new run), the request what for (its inversion too).
    assert went_on.asked == {"process", "invert"} and went_on.redo and went_on.option == 5
    assert (went_on.positions_m, went_on.workers) == ([30.0], 10)
    # An option of a later stage: the request's stages from it on, none before.
    curves = Offer("pick every window again", 'pick(run_id="r", windows="all")')
    assert continued(_scope(pick=True, option=2), request, curves).asked == {"pick", "invert"}
    models = Offer("invert again", 'redo(run_id="r", stage="inversion")')
    assert continued(_scope(option=1), request, models).asked == {"invert"}
    # A run to look at: nothing more of the request, what it asked being there.
    look = Offer("work on run r (active, images)", 'inspect(what="run", run_id="r")')
    assert continued(_scope(option=2), _scope(process=True), look).asked == frozenset()


def test_a_call_on_another_run_than_the_one_named_is_refused() -> None:
    named = _scope(invert=True, run_id="20990101-000000-abcd")
    latest = '{"run_id": "20261004-075517-daeb"}'

    assert refusal(named, "invert", '{"run_id": "20990101-000000-abcd"}') is None
    assert refusal(named, "invert", latest) == (
        "Not called: this message names run 20990101-000000-abcd, and invert is on run "
        "20261004-075517-daeb: work on the run it names. If that run does not exist, say so, "
        "with the runs there (inspect lists them), and ask which one the user means."
    )
    # The run the turn made, or the one the option chosen names: worked on.
    assert refusal(named, "invert", latest, frozenset({"20261004-075517-daeb"})) is None
    # Reading another run, always.
    assert refusal(named, "inspect", '{"what": "run", "run_id": "20261004-075517-daeb"}') is None


def test_a_message_of_signs_asks_nothing_and_chooses_nothing() -> None:
    # A typo while options wait ("%"): no answer to them, the model not asked.
    model = FormModel()
    offered = Context(offers=(Offer("a new run", 'run_processing(profile="p", again=true)'),))

    read = anyio.run(read_scope, model, "%", offered)

    assert read.scope == _scope() and read.fills == () and model.sent == []
