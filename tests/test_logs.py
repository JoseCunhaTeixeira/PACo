"""Log lines as JSON, each naming the conversation, turn and run it belongs to."""

import json
import logging

from paco import logs


def test_a_log_line_is_json_with_what_it_belongs_to() -> None:
    record = logging.LogRecord(
        "paco.server", logging.WARNING, __file__, 1, "No trace of %s", ("pick",), None
    )
    conversation, turn = logs.CONVERSATION.set("c1"), logs.TURN.set(2)
    run = logs.RUN.set("20260930-161253-89f5")
    try:
        line = json.loads(logs.JsonLines().format(record))
    finally:
        logs.CONVERSATION.reset(conversation)
        logs.TURN.reset(turn)
        logs.RUN.reset(run)

    assert {
        key: line[key] for key in ("level", "logger", "message", "conversation", "turn", "run")
    } == {
        "level": "WARNING",
        "logger": "paco.server",
        "message": "No trace of pick",
        "conversation": "c1",
        "turn": 2,
        "run": "20260930-161253-89f5",
    }
    assert line["at"].endswith("+00:00")
    # Outside a call, the line names nothing it does not belong to.
    assert set(json.loads(logs.JsonLines().format(record))) == {"at", "level", "logger", "message"}
