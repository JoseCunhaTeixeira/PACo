"""The objective (docs/autonomous_plan.md §4.0): PACo's scope form, subclassed with the objective's four
fields and the settings a user types beyond the windows' (the images' velocities and band, the
inversion's iterations and Vs bounds), read from the user's message by the model (the form's JSON
schema constrains its output) and checked in code, as PACo reads its scope: its prompt and worked
examples, with the objective's own after them. The settings the user gives in it stay as they
gave them (§8, rule 1; paco.autonomous.tools.given)."""

import json
from typing import Any, Literal, Self

from openai.types.chat import ChatCompletionMessageParam
from pydantic import Field, ValidationError, model_validator

from paco import prompts as paco_prompts
from paco.agent.model import ChatModel, Filled
from paco.agent.scope import Context, Read, Scope, ScopeError, checked
from paco.autonomous import prompts
from paco.autonomous.tools.given import BOUNDS, ITERATIONS, Given


class GeoScope(Scope):
    """PACo's scope, with the objective: the depth to reach, what comes first, the effort, and
    what the user says about the ground (every field required, as PACo's: the schema makes the
    model write each)."""

    target_depth_m: float | None = Field(
        gt=0, description="The depth the message asks the models to reach, in metres."
    )
    priority: Literal["depth", "lateral", "balanced"] | None = Field(
        description="What it puts first: depth, lateral detail, or a balance."
    )
    effort: Literal["quick", "normal", "thorough"] | None = Field(
        description="How much work it asks."
    )
    site_notes: str | None = Field(
        max_length=200, description="What it says about the ground, in its words."
    )
    vmin_m_s: float | None = Field(
        gt=0, description="The images' lowest phase velocity it gives, in m/s."
    )
    vmax_m_s: float | None = Field(
        gt=0, description="The images' highest phase velocity it gives, in m/s."
    )
    fmin_hz: float | None = Field(ge=0, description="The images' lowest frequency it gives, in Hz.")
    fmax_hz: float | None = Field(
        gt=0, description="The images' highest frequency it gives, in Hz."
    )
    iterations: int | None = Field(
        gt=0, description="The inversion's iterations it gives, each chain's."
    )
    vs_min_m_s: float | None = Field(
        gt=0, description="The models' lowest shear-wave velocity it allows, in m/s."
    )
    vs_max_m_s: float | None = Field(
        gt=0, description="The models' highest shear-wave velocity it allows, in m/s."
    )

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        """Each range low then high."""
        for low, high in (
            ("vmin_m_s", "vmax_m_s"),
            ("fmin_hz", "fmax_hz"),
            ("vs_min_m_s", "vs_max_m_s"),
        ):
            first, last = getattr(self, low), getattr(self, high)
            if first is not None and last is not None and first >= last:
                raise ValueError(f"{low} ({first:g}) must be below {high} ({last:g})")
        return self

    def objective(self) -> str:
        """The objective in a line, as the journal's first entry says it."""
        priority = {"depth": "depth first", "lateral": "lateral detail first"}.get(
            self.priority or "", "balanced"
        )
        depth = (
            f"down to {self.target_depth_m:g} m"
            if self.target_depth_m is not None
            else "no target depth"
        )
        said = f"{priority}, {depth}, {self.effort or 'normal'} effort"
        return said + (f"; the ground: {self.site_notes}" if self.site_notes else "")

    def objective_fields(self, spacing_m: float) -> dict[str, Any]:
        """The objective's fields, and the profile and the settings the message gives, on a line
        whose receivers are `spacing_m` apart."""
        return {
            "target_depth_m": self.target_depth_m,
            "priority": self.priority or "balanced",
            "effort": self.effort or "normal",
            "site_notes": self.site_notes,
            "profile": self.profile,
            "given": self.given(spacing_m).model_dump(mode="json"),
        }

    def given(self, spacing_m: float) -> Given:
        """The settings the message gives: the line's as commit's changes take them (the windows'
        length in receivers, one given in metres converted on a line whose receivers are
        `spacing_m` apart; the images' velocities and band), and the inversion's (its iterations,
        every layer's Vs bounds)."""
        line: dict[str, dict[str, Any]] = {}
        if self.length_receivers is not None:
            line["masw"] = {"length": self.length_receivers}
        elif self.length_m is not None:
            line["masw"] = {"length": round(self.length_m / spacing_m) + 1}
        images = {
            key: value
            for key, value in (
                ("vmin", self.vmin_m_s),
                ("vmax", self.vmax_m_s),
                ("fmin", self.fmin_hz),
                ("fmax", self.fmax_hz),
            )
            if value is not None
        }
        if images:
            line["dispersion"] = images
        inversion = {
            key: value
            for key, value in zip(
                (ITERATIONS, *BOUNDS),
                (self.iterations, self.vs_min_m_s, self.vs_max_m_s),
                strict=True,
            )
            if value is not None
        }
        return Given(mode=self.mode, line=line, inversion=inversion)

    def line(self) -> str:
        """PACo's scope line, with the objective."""
        return f"{super().line().removesuffix('.')} · {self.objective()}."


# The form's JSON schema, as the model's output is constrained to it.
SCHEMA: dict[str, Any] = GeoScope.model_json_schema()
# A message asking nothing: the worked examples say what differs from it.
BLANK: dict[str, Any] = {
    name: [] if name in ("positions_m", "compare_lengths_receivers", "compare_lengths_m") else None
    for name in GeoScope.model_fields
} | dict.fromkeys(("process", "pick", "invert", "soils", "redo", "replace_hand_work"), False)


async def read_objective(model: ChatModel, message: str, context: Context) -> Read:
    """The scope and the objective of `message`: the model fills the form, strictly parsed; a
    form that does not parse is sent back once with its error. Raises ScopeError when the second
    does not parse either."""
    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": f"{paco_prompts.prompt('scope')}\n\n{prompts.prompt('objective')}",
        },
        *_examples(),
        {"role": "user", "content": _with_context(message, context)},
    ]
    fills: list[Filled] = []
    said = ""
    for _ in range(2):
        filled = await model.fill(messages, SCHEMA)
        fills.append(filled)
        try:
            scope = GeoScope.model_validate_json(filled.content)
        except ValidationError as error:
            said = "; ".join(
                f"{'.'.join(str(part) for part in item['loc']) or 'form'}: {item['msg']}"
                for item in error.errors()
            )
            messages.append({"role": "assistant", "content": filled.content})
            messages.append({"role": "user", "content": f"Not valid: {said}. Fill the form again."})
            continue
        return Read(scope=checked(scope, context, message), fills=tuple(fills))
    raise ScopeError(said)


def _examples() -> list[ChatCompletionMessageParam]:
    """The worked examples, PACo's then the objective's, as earlier messages and their forms."""
    turns: list[ChatCompletionMessageParam] = []
    for message, form in (*paco_prompts.scope_examples(), *prompts.objective_examples()):
        turns.append({"role": "user", "content": f"Message: {message}"})
        turns.append({"role": "assistant", "content": json.dumps({**BLANK, **form})})
    return turns


def _with_context(message: str, context: Context) -> str:
    """The message, after the current profile and run, as PACo's form reads it."""
    lines: list[str] = []
    if context.profile or context.run_id:
        lines.append(f"Current profile: {context.profile}; current run: {context.run_id}.")
    lines.append(f"Message: {message}")
    return "\n".join(lines)
