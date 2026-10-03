"""The answer the user reads: the model's text in its form, around what the tools did; the
question shown only when the turn leaves the user a choice."""

import json
from typing import Any

import anyio
import pytest
from openai.types.chat import ChatCompletionFunctionToolParam, ChatCompletionMessageParam
from pydantic import ValidationError

from paco import prompts
from paco.agent.answer import (
    SCHEMA,
    AnswerError,
    AnswerForm,
    Turn,
    render,
    unfound_numbers,
    write_answer,
)
from paco.agent.model import Filled, Reply
from paco.agent.scope import Scope

ASKS_PICK = Scope(
    process=False,
    pick=True,
    invert=False,
    soils=False,
    profile="active_p1",
    run_id=None,
    positions_m=[],
    length_receivers=None,
    length_m=None,
    step_receivers=None,
    step_m=None,
    compare_lengths_receivers=[],
    compare_lengths_m=[],
    redo=False,
    replace_hand_work=False,
    option=None,
)
PICKED = json.dumps(
    {
        "run_id": "r",
        "status": "partial",
        "did": "Picked run r, 4 windows: G3 3 pass, 1 reject.",
        "left": ["xmid 2.88 (1): G3 narrow_span"],
        "used": ["picking: M0 tracked along its ridge"],
        "changed": ["min_relative_coherence 0.18 -> 0.11 at xmid 2.88 (1), by G3:narrow_span"],
        "summary": "G3: 3 pass, 1 reject; the curves span 5.2 to 12.46 m.",
        "next": "",
    }
)
OFFERED = json.dumps(
    {
        "run_id": "r",
        "status": "refused",
        "summary": "Run r holds 4 curves.",
        "next": "Nothing was done.",
        "options": [
            {"label": "pick every window again", "call": 'pick(run_id="r", windows="all")'},
            {"label": "invert the curves as they are", "call": 'invert(run_id="r")'},
        ],
    }
)


class FormModel:
    """Stands in for the model filling answer forms: its fills in order, what it was sent."""

    def __init__(self, *fills: str) -> None:
        self._fills = list(fills)
        self.sent: list[list[ChatCompletionMessageParam]] = []

    async def __call__(
        self,
        messages: list[ChatCompletionMessageParam],  # noqa: ARG002
        tools: list[ChatCompletionFunctionToolParam],  # noqa: ARG002
    ) -> Reply:
        raise AssertionError("an answer form, not a conversation")

    async def fill(
        self, messages: list[ChatCompletionMessageParam], schema: dict[str, Any]
    ) -> Filled:
        assert schema == SCHEMA
        self.sent.append(list(messages))
        return Filled(content=self._fills.pop(0))


def test_the_text_can_hold_no_question_mark() -> None:
    assert AnswerForm(said="Picked 4 windows.", question=None).said == "Picked 4 windows."
    for asking in ("Shall I invert them?", "Voulez-vous la suite" + chr(0xFF1F)):
        with pytest.raises(ValidationError):
            AnswerForm(said=asking, question=None)
    assert SCHEMA["properties"]["said"]["pattern"] == "^[^?\\uff1f]*$"


def test_the_model_fills_the_form_from_its_draft() -> None:
    form = {"said": "Picked 4 windows.", "question": None}
    model = FormModel('{"said": 1}', json.dumps(form))

    written = anyio.run(write_answer, model, "Pick active_p1.", "Picked 4 windows. Shall I?")

    assert written.form == AnswerForm.model_validate(form) and len(written.fills) == 2
    system, *examples, user = model.sent[0]
    assert system == {"role": "system", "content": prompts.prompt("answer")}
    assert len(examples) == 2 * len(prompts.answer_examples())
    assert user.get("content") == "Message: Pick active_p1.\nAnswer: Picked 4 windows. Shall I?"
    with pytest.raises(AnswerError):
        anyio.run(write_answer, FormModel("{}", "{}"), "Pick active_p1.", "Picked.")


def test_the_turn_reads_what_each_tool_did() -> None:
    turn = Turn()

    turn.read("pick", PICKED, failed=False)
    turn.read("pick", "Error executing tool pick: [precondition] Unknown run 'x'.", failed=True)

    assert turn.done == ["Picked run r, 4 windows: G3 3 pass, 1 reject."]
    assert turn.left == ["xmid 2.88 (1): G3 narrow_span"]
    assert turn.stages == ["curves"] and turn.worked and not turn.stuck and not turn.offers
    turn.read("redo", "Error executing tool redo: [stuck] The retry budget is spent.", True)
    assert turn.stuck


def test_after_work_the_answer_says_what_next_not_the_models_question() -> None:
    turn = Turn()
    turn.read("pick", PICKED, failed=False)

    answer = render(ASKS_PICK, "Picked active_p1's 4 windows.", "Invert them?", turn)

    assert answer == (
        "Scope: pick · active_p1.\n\n"
        "Picked active_p1's 4 windows.\n\n"
        "Done:\n- Picked run r, 4 windows: G3 3 pass, 1 reject.\n\n"
        "Left out:\n- xmid 2.88 (1): G3 narrow_span\n\n"
        "Next: the Vs models (ask to invert them); the results in PAC's Visualization page.\n\n"
        "Parameters used:\n- picking: M0 tracked along its ridge\n\n"
        "Settings the gates changed:\n"
        "- min_relative_coherence 0.18 -> 0.11 at xmid 2.88 (1), by G3:narrow_span"
    )


def test_the_question_shows_when_the_user_must_choose() -> None:
    offered = Turn()
    offered.read("pick", OFFERED, failed=False)
    stuck = Turn()
    stuck.read("redo", "Error executing tool redo: [stuck] The budget is spent.", True)
    nothing = Turn()

    with_options = render(ASKS_PICK, "Run r holds 4 curves.", None, offered)
    when_stuck = render(ASKS_PICK, "The budget is spent.", "A new run, or stop?", stuck)
    before_work = render(ASKS_PICK, "Two runs hold active_p1.", "Which one?", nothing)

    assert with_options.endswith(
        "Which do you choose?\n(1) pick every window again\n(2) invert the curves as they are"
    )
    assert when_stuck.endswith("A new run, or stop?")
    assert before_work.endswith("Which one?")
    # A message asking for no work (a 3D model no tool makes) gets an answer, not a question.
    nothing_asked = ASKS_PICK.model_copy(update={"pick": False})
    impossible = render(
        nothing_asked, "No tool makes a 3D model.", "A 2D section instead?", nothing
    )
    assert impossible.endswith("No tool makes a 3D model.")


def test_numbers_no_result_holds_are_flagged() -> None:
    turn = Turn()
    turn.read("pick", PICKED, failed=False)

    answer = render(ASKS_PICK, "The curves span 5.2 to 12.5 m, and reach 40 m.", None, turn)

    assert "(Not in this turn's results: 40.)" in answer
    assert unfound_numbers("12,5 m and 3 windows", ["12.46"]) == []
    assert unfound_numbers("12.4 m", ["12.46"]) == ["12.4"]


def test_windows_left_out_over_a_setting_given_are_named_next() -> None:
    turn = Turn()
    turn.read(
        "pick",
        PICKED.replace(
            "G3 narrow_span", "G2 locked, asks dispersion vmax 900 (given: 250), ridge_at_vmax"
        ),
        failed=False,
    )

    answer = render(ASKS_PICK, "Picked active_p1's 4 windows.", None, turn)

    assert (
        "Next: for the windows left out over a setting you gave, ask again with the change; the "
        "Vs models (ask to invert them); the results in PAC's Visualization page." in answer
    )


def test_a_stage_asked_that_no_tool_did_is_said() -> None:
    asks = ASKS_PICK.model_copy(update={"invert": True})
    picked = Turn()
    picked.read("pick", PICKED, failed=False)
    offered = Turn()
    offered.read("pick", OFFERED, failed=False)
    nothing = Turn()

    assert "Asked but not done: invert." in render(asks, "Picked the 4 windows.", None, picked)
    # Nothing run and nothing asked back: the answer says what was not done.
    nothing_done = render(asks, "The run holds images.", None, nothing)
    assert "Asked but not done: pick, invert." in nothing_done
    # When the user must choose, the question says why.
    assert "Asked but not done" not in render(asks, "Run r holds 4 curves.", None, offered)
    assert "Asked but not done" not in render(asks, "Two runs hold it.", "Which one?", nothing)


def test_a_comparison_or_a_redo_does_the_stage_it_serves() -> None:
    process = ASKS_PICK.model_copy(update={"process": True, "pick": False})
    compared = Turn()
    compared.read("compare", json.dumps({"did": "Compared 2 variants of active_p1."}), False)
    redone = Turn()
    did = {"run_id": "r", "status": "ok", "did": "Redid the phase shift of run r, 1 window."}
    redone.read("redo", json.dumps(did), False, '{"run_id": "r", "stage": "phase_shift"}')

    assert "Asked but not done" not in render(process, "Variant 2 reaches deeper.", None, compared)
    assert "Asked but not done" not in render(process, "Redone.", None, redone)
    # The images made again: the curves come next.
    assert redone.stages == ["images"]
