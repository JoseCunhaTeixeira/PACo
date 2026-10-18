"""The journal (docs/geo_plan.md §7.1): the model's entries, their limits and ids, and the
readings."""

from pathlib import Path

import pytest

from paco.geo.tools import journal
from paco.geo.tools.trials import Study


@pytest.fixture
def study(tmp_path: Path) -> Study:
    folder = tmp_path / "studies" / "20261017-120000-abcd"
    folder.mkdir(parents=True)
    return Study("active_p2", folder)


def test_the_models_entries_take_their_kinds_ids(study: Study) -> None:
    journal.objective(study, {"priority": "balanced"}, "balanced, no target depth")
    hypothesis = journal.write(
        study, "hypothesis", "The near field  biases\nthe long wavelengths.", expect="+3 windows"
    )
    conclusion = journal.write(study, "conclusion", "Not worth it.", of=["H1"])

    assert (hypothesis.id, hypothesis.text) == ("H1", "The near field biases the long wavelengths.")
    assert conclusion.id == "C1"
    assert [entry.id for entry in journal.entries(study)] == ["O1", "H1", "C1"]
    assert journal.read(study).splitlines() == [
        "O1 balanced, no target depth",
        "H1 The near field biases the long wavelengths.; expects: +3 windows",
        "C1 Not worth it. (of H1)",
    ]
    assert '"expect": "+3 windows"' in journal.read(study, entry="H1")


@pytest.mark.parametrize(
    ("kind", "text", "of", "expect", "said"),
    [
        ("decision", "", (), None, "one paragraph"),
        ("decision", "x" * 601, (), None, "600 characters at most"),
        ("conclusion", "Fine.", ("E9",), None, "no entry E9"),
        ("hypothesis", "Longer windows reach deeper.", (), None, "expect"),
    ],
)
def test_an_entry_is_refused_saying_why(
    study: Study,
    kind: journal.ModelKind,
    text: str,
    of: tuple[str, ...],
    expect: str | None,
    said: str,
) -> None:
    with pytest.raises(ValueError, match=said):
        journal.write(study, kind, text, of, expect)
    assert journal.entries(study) == []


def test_the_readings_of_a_setting(study: Study) -> None:
    journal.append(
        study,
        journal.Entry(
            id="E1",
            kind="experiment",
            at=journal.datetime.now(journal.UTC),
            text="on spread:9: current 5/9; 1 {...} 7/9 (+2, -0)",
            data={"variants": [{"changes": {}}, {"changes": {"masw": {"length": 13}}}]},
        ),
    )

    assert journal.read(study, setting="masw length").startswith("E1 on spread:9")
    assert journal.read(study, setting="masw").startswith("E1 on spread:9")
    assert journal.read(study, setting="muting") == "No experiment or commit changed muting."
    with pytest.raises(ValueError, match="No entry E2"):
        journal.read(study, entry="E2")
