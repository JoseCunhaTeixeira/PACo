"""The flags' classes (docs/autonomous_plan.md §5.2): every flag PACo's gates G1 to G6 can raise has its
classes, and each flag's class is read from its action."""

import ast
from importlib.util import find_spec
from pathlib import Path

import pytest
from autonomous_judged import JudgedLine

from paco.autonomous.sensors.flags import FLAGS, FlagClass, flag_class
from paco.qc.models import Action, ExcludeRecord, ExcludeTraces, Flag, Keep, Override, Reject

# The functions of PACo's gates that make a flag of the name they are given: G2's _short and the
# change of G5's loop.
NAMED_BY_CALLER = ("_short", "change")


def test_every_flag_the_gates_can_raise_has_its_classes() -> None:
    # Read from PACo's code: a flag a new PACo adds fails here until it has its classes.
    raised, unresolved = _raised_flags()

    assert unresolved == []
    assert raised == set(FLAGS)


@pytest.mark.parametrize(
    ("action", "fixable", "expected"),
    [
        (Reject(reason="the ridge barely stands out"), False, "blocking"),
        (Override(stage="picking", overrides={"fmax": 37.0}), True, "fixable"),
        # G5's no_mode: a pick for the user, its model rejected.
        (Override(stage="picking", overrides={"fmax": 18.0}), False, "blocking"),
        (Keep(note="near shots only"), True, "note"),
        (ExcludeTraces(record="2.dat", traces=(1, 89)), False, "automatic"),
        (ExcludeRecord(record="2.dat"), False, "automatic"),
    ],
)
def test_a_flags_class_comes_from_its_action(
    action: Action, fixable: bool, expected: FlagClass
) -> None:
    flag = Flag(
        name="a_flag",
        message="What the gate says.",
        stage="picking",
        action=action,
        fixable=fixable,
    )

    assert flag_class(flag) == expected


@pytest.mark.slow
def test_the_lines_flags_take_the_classes_the_table_gives(judged_line: JudgedLine) -> None:
    raised = [
        (gate, flag)
        for attempt in judged_line.attempts
        for gate, result in attempt.results.items()
        for flag in result.flags
    ]

    assert {flag.name for _, flag in raised} >= {"near_field", "outlier"}
    for gate, flag in raised:
        assert flag_class(flag) in FLAGS[(gate, flag.name)], (gate, flag.name)


def _raised_flags() -> tuple[set[tuple[str, str]], list[str]]:
    """Every (gate, flag) the modules of G1 to G6 raise, read from their code: Flag(name=...) with
    a name, or with a loop's name over a tuple of names, and the names given to the functions
    that make a flag of them. The Flag(...) calls none of these names come back unresolved."""
    spec = find_spec("paco.qc.models")
    assert spec is not None and spec.origin is not None
    raised: set[tuple[str, str]] = set()
    unresolved: list[str] = []
    for path in sorted(Path(spec.origin).parent.glob("g[1-6]_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        gate = _gate(tree)
        looped = _looped_names(tree)
        named_by_caller = _lines_of(tree, NAMED_BY_CALLER)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            called = _called(node)
            if called == "Flag":
                given = next((word.value for word in node.keywords if word.arg == "name"), None)
                if (name := _text(given)) is not None:
                    raised.add((gate, name))
                elif node.lineno in looped:
                    raised.update((gate, name) for name in looped[node.lineno])
                elif node.lineno not in named_by_caller:
                    unresolved.append(f"{path.name}:{node.lineno}")
            elif called in NAMED_BY_CALLER and node.args:
                if (name := _text(node.args[0])) is not None:
                    raised.add((gate, name))
    return raised, unresolved


def _gate(tree: ast.Module) -> str:
    """The module's GATE."""
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "GATE" for target in node.targets
        ):
            gate = _text(node.value)
            assert gate is not None
            return gate
    raise AssertionError("a gate's module without its GATE")


def _looped_names(tree: ast.Module) -> dict[int, set[str]]:
    """For each call inside a `for name, ... in (("a", ...), ...)` loop, by its line: the names
    the loop goes through."""
    found: dict[int, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.For) or not isinstance(node.iter, ast.Tuple):
            continue
        first = node.target.elts[0] if isinstance(node.target, ast.Tuple) else node.target
        if not (isinstance(first, ast.Name) and first.id == "name"):
            continue
        names = {
            name
            for item in node.iter.elts
            if (name := _text(item.elts[0] if isinstance(item, ast.Tuple) else item)) is not None
        }
        for inner in ast.walk(node):
            if isinstance(inner, ast.Call):
                found[inner.lineno] = names
    return found


def _lines_of(tree: ast.Module, functions: tuple[str, ...]) -> set[int]:
    """The lines of the functions (and methods) named `functions`."""
    return {
        line
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in functions
        for line in range(node.lineno, (node.end_lineno or node.lineno) + 1)
    }


def _called(node: ast.Call) -> str:
    """The name of the function `node` calls."""
    if isinstance(node.func, ast.Name):
        return node.func.id
    return node.func.attr if isinstance(node.func, ast.Attribute) else ""


def _text(node: ast.expr | None) -> str | None:
    """`node`'s string, when it is a string constant."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None
