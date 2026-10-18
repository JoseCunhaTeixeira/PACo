"""The judge (docs/geo_plan.md §12): a model grades the geophysicist agent's play from 1 to 5, its
reply read as PACo reads its judge's (paco.evaluation.judging). Two grades: the answer against the
scenario's rubric, with PACo's instructions, the play shown from its steps (its conversation is
rebuilt at the freeze, the steps keep every call); and the reasoning against the journal's rubric:
are the hypotheses supported, the conclusions faithful to the numbers."""

from collections.abc import Sequence

from paco.agent.model import ChatModel
from paco.agent.record import Transcript
from paco.evaluation.judging import INSTRUCTIONS, read_score
from paco.evaluation.models import JudgeScore
from paco.geo.tools import journal
from paco.geo.tools.trials import Study

_RESULT_LENGTH = 300  # characters of each tool result shown to the judge, as PACo's

JOURNAL_INSTRUCTIONS = (
    "You grade the reasoning of an agent that processes an MASW seismic line as a geophysicist, "
    "from its journal. Read the rubric and the journal, then grade the reasoning from 1 "
    "(conclusions that contradict or invent the numbers) to 5 (every step supported and faithful "
    'to them). Reply with JSON only: {"score": <1 to 5>, "reason": "<one sentence>"}.'
)
# What the reasoning is graded on (§12, "Reasoning").
JOURNAL_RUBRIC = (
    "The journal's entries, in order: the objective (O), the survey (S), hypotheses (H, each with "
    "what it expects), experiments (E, numbers code wrote: each variant's windows passing out of "
    "those tried, gained and lost), conclusions (C), commits of the line (L), decisions (D), the "
    "freeze (F), inversions (I, G5's verdicts) and the report (R). Each hypothesis rests on an "
    "observation and says what it expects; each experiment tests one; each conclusion says what "
    "the experiment's numbers show, no window, depth or share they do not; the commits and "
    "decisions follow from the conclusions; the trade-offs cite the objective."
)


async def judge_answer(
    model: ChatModel, rubric: str, questions: Sequence[str], transcript: Transcript
) -> JudgeScore:
    """The judge's grade of the play's answer against `rubric`, PACo's instructions."""
    prompt = f"Expectation: {rubric}\n\nConversation:\n{render_play(questions, transcript)}"
    reply = await model(
        [{"role": "system", "content": INSTRUCTIONS}, {"role": "user", "content": prompt}], []
    )
    return read_score(reply.content)


async def judge_journal(model: ChatModel, study: Study) -> JudgeScore:
    """The judge's grade of the reasoning in `study`'s journal."""
    entries = journal.entries(study)
    text = "\n".join(journal.line(entry) for entry in entries) or "(empty)"
    prompt = f"Rubric: {JOURNAL_RUBRIC}\n\nJournal:\n{text}"
    reply = await model(
        [{"role": "system", "content": JOURNAL_INSTRUCTIONS}, {"role": "user", "content": prompt}],
        [],
    )
    return read_score(reply.content)


def render_play(questions: Sequence[str], transcript: Transcript) -> str:
    """The play as the judge reads it: the user's messages, every call the model asked for and
    its result (cut short, as PACo's judge reads them), the host's refusals, and the answer."""
    lines = [f"USER: {question}" for question in questions]
    for step in transcript.tool_steps:
        lines.append(f"TOOL CALL: {step.name} {step.arguments}")
        result = step.result
        cut = result if len(result) <= _RESULT_LENGTH else result[:_RESULT_LENGTH] + "..."
        lines.append(f"{'TOOL RESULT' if step.called else 'REFUSED'}: {cut}")
    if transcript.answer:
        lines.append(f"ASSISTANT: {transcript.answer}")
    return "\n".join(lines)
