"""PACo's atlas: reference figures of good and bad data, each sheet's
caption read as a prompt is; drawn again from PAC's test lines, the same figures."""

from pathlib import Path

import pytest
from local_lines import ACTIVE, LOCAL_INPUT_DIR, PASSIVE, present

from paco.autonomous import prompts
from paco.autonomous.atlas import CORRELATIONS, IMAGES, make


def test_each_sheet_has_its_caption_and_figure() -> None:
    sheets: dict[str, tuple[str, bytes]] = dict(zip(prompts.ATLAS, prompts.atlas(), strict=True))

    for name, (caption, figure) in sheets.items():
        assert caption.startswith("PACo's atlas, ") and name in caption.splitlines()[0]
        assert figure.startswith(b"\x89PNG")
        # A paragraph or a list item on one line, as the model reads it.
        assert all(len(line) > 60 for line in caption.splitlines() if line), name
    # Each panel the figure titles, the caption names, in order.
    for name, examples in (("images", IMAGES), ("correlations", CORRELATIONS)):
        items = [line for line in sheets[name][0].splitlines() if line.startswith("- ")]
        assert [item[2:].split(" (")[0] for item in items] == [one.title for one in examples]


@pytest.mark.slow
def test_the_atlas_is_drawn_again_alike(tmp_path: Path) -> None:
    # A change of the processing or the drawing that moves a figure asks to draw it again, to
    # read its caption again, and a new prompts' version.
    present(ACTIVE, PASSIVE)

    make(LOCAL_INPUT_DIR, tmp_path / "scratch", tmp_path / "atlas")

    for name in prompts.ATLAS:
        drawn = (tmp_path / "atlas" / f"{name}.png").read_bytes()
        assert drawn == (prompts.ATLAS_FOLDER / f"{name}.png").read_bytes(), name
