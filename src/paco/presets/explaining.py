"""Validation errors rewritten for the agent: one line per problem, saying what to send instead."""

import difflib
import json
import typing
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ValidationError
from pydantic_core import ErrorDetails

_BOUNDS = {
    "greater_than": (">", "gt"),
    "greater_than_equal": (">=", "ge"),
    "less_than": ("<", "lt"),
    "less_than_equal": ("<=", "le"),
}
_TYPES = {
    "int_parsing": "an integer",
    "int_from_float": "an integer",
    "int_type": "an integer",
    "float_parsing": "a number",
    "float_type": "a number",
    "bool_parsing": "true or false",
    "bool_type": "true or false",
    "string_type": "a string",
}
_NOT_AN_OBJECT = {"model_type", "model_attributes_type", "dict_type"}
# Keys no agent sets: `mode` comes from the preset name, `method` is listed as a method.
_HIDDEN_KEYS = {"mode", "method"}


def explain(error: ValidationError, name: str, presets: Mapping[str, type[BaseModel]]) -> str:
    """`error`, raised while validating overrides for preset `name`, as the agent reads it."""
    lines = [_explain(details, name, presets) for details in error.errors()]
    return f"Invalid overrides for preset '{name}':\n" + "\n".join(f"- {line}" for line in lines)


def _explain(error: ErrorDetails, name: str, presets: Mapping[str, type[BaseModel]]) -> str:
    preset = presets[name]
    path, parent = _locate(preset, error["loc"])
    key = path.rpartition(".")[2]
    field = parent.model_fields.get(key) if parent is not None else None
    context: dict[str, Any] = error.get("ctx", {})
    kind = error["type"]
    got = _show(error["input"])

    if kind == "extra_forbidden":
        return _unknown_key(path, key, parent, name, presets)
    if kind in ("union_tag_invalid", "union_tag_not_found") and field is not None:
        methods = list(_methods(field.annotation))
        if kind == "union_tag_not_found":
            return f'{path}: missing "method". Methods: {", ".join(methods)}.'
        tag = str(context["tag"])
        return (
            f"{path}.method: unknown method '{tag}'. Allowed: {', '.join(methods)}."
            f"{_did_you_mean(tag, methods)}"
        )
    if kind in _BOUNDS:
        sign, bound = _BOUNDS[kind]
        return f"{path}: must be {sign} {context[bound]:g} (got {got})."
    if kind in _TYPES:
        return f"{path}: must be {_TYPES[kind]} (got {got})."
    if kind in _NOT_AN_OBJECT:
        return f"{path}: must be an object (got {got}).{_contents(field)}"
    if kind == "literal_error" and key == "mode":
        return (
            f"mode: set by the preset name ('{preset.model_fields['mode'].default}'); leave it out."
        )
    if kind == "missing":
        return f"{path}: required."
    if kind == "value_error":
        return f"{path}: {error['msg'].removeprefix('Value error, ')}."
    return f"{path}: {error['msg']}."


def _unknown_key(
    path: str,
    key: str,
    parent: type[BaseModel] | None,
    name: str,
    presets: Mapping[str, type[BaseModel]],
) -> str:
    fields = parent.model_fields if parent is not None else {}
    allowed = [field for field in fields if field not in _HIDDEN_KEYS]
    if parent is presets[name]:
        # A stage of another preset: say so, rather than suggest a look-alike name.
        if owners := [other for other, preset in presets.items() if key in preset.model_fields]:
            return (
                f"{path}: not a stage of preset '{name}', only of {', '.join(owners)}. "
                f"Stages: {', '.join(allowed)}."
            )
        return f"{path}: unknown stage. Allowed: {', '.join(allowed)}.{_did_you_mean(key, allowed)}"
    if not allowed and parent is not None:
        method = parent.model_fields["method"].default
        return f"{path}: unknown parameter; method '{method}' takes no parameters."
    return f"{path}: unknown parameter. Allowed: {', '.join(allowed)}.{_did_you_mean(key, allowed)}"


def _locate(
    preset: type[BaseModel], loc: tuple[int | str, ...]
) -> tuple[str, type[BaseModel] | None]:
    """The path as the agent wrote it, and the model its last key belongs to.

    pydantic's locations include the method of a stage with a choice of methods
    (`filtering.iir.fmaxx`); the agent never wrote that part, so it is left out of the path.
    """
    parts = [str(part) for part in loc]
    keys: list[str] = []
    owner: type[BaseModel] | None = preset
    parent: type[BaseModel] | None = preset
    index = 0
    while index < len(parts):
        key = parts[index]
        keys.append(key)
        parent = owner
        index += 1
        field = owner.model_fields.get(key) if owner is not None else None
        methods = _methods(field.annotation) if field is not None else {}
        if index < len(parts) and parts[index] in methods:
            owner = methods[parts[index]]
            index += 1
        elif field is not None and _is_model(field.annotation):
            owner = field.annotation
        else:
            owner = None
    return ".".join(keys), parent


def _methods(annotation: Any) -> dict[str, type[BaseModel]]:  # noqa: ANN401
    """The models of a stage with a choice of methods, by method name; empty for anything else."""
    return {
        member.model_fields["method"].default: member
        for member in typing.get_args(annotation)
        if _is_model(member) and "method" in member.model_fields
    }


def _contents(field: Any) -> str:  # noqa: ANN401
    """What an object at `field` holds: its methods, or its parameters."""
    if field is None:
        return ""
    if methods := _methods(field.annotation):
        return f" Methods: {', '.join(methods)}."
    if _is_model(field.annotation):
        names = [name for name in field.annotation.model_fields if name not in _HIDDEN_KEYS]
        return f" Parameters: {', '.join(names)}."
    return ""


def _did_you_mean(word: str, options: list[str]) -> str:
    matches = difflib.get_close_matches(word, options, n=1, cutoff=0.6)
    return f" Did you mean {matches[0]}?" if matches else ""


def _show(value: Any) -> str:  # noqa: ANN401
    text = json.dumps(value) if isinstance(value, dict | list) else repr(value)
    return text if len(text) <= 60 else text[:57] + "..."


def _is_model(annotation: Any) -> bool:  # noqa: ANN401
    return isinstance(annotation, type) and issubclass(annotation, BaseModel)
