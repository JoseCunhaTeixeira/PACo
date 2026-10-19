"""observe, look and advise (docs/autonomous_plan.md §6): what they read of a committed line, here
active_test_profile's four 24-receiver windows as PACo makes them; and advise's own trials, made once."""

from pathlib import Path

import pytest
from autonomous_judged import JudgedLine

from paco.autonomous.tools.advise import advise
from paco.autonomous.tools.look import look
from paco.autonomous.tools.observe import observe
from paco.autonomous.tools.trials import Study, new_study
from paco.qc.config import QCConfig
from paco.settings import Settings

PNG = b"\x89PNG"


@pytest.mark.slow
def test_observe_summarizes_a_gate(judged_line: JudgedLine) -> None:
    text = observe(judged_line.folder, "curves", QCConfig())

    assert text.splitlines()[0] == "G3: 3 pass, 1 reject"
    assert "G3 near_field, xmid 17.25-125.25 (4)" in text


@pytest.mark.slow
def test_observe_gives_the_rows_where_asked(judged_line: JudgedLine) -> None:
    lines = observe(
        judged_line.folder, "curves", QCConfig(), where="flag:near_field+xmids:80-130"
    ).splitlines()

    assert lines[0] == "G3 where flag:near_field+xmids:80-130: 2 units, page 1 of 1."
    assert [line.split(":")[0] for line in lines[1:]] == ["89.25", "125.25"]
    assert lines[1].startswith("89.25: pass; near_field; λ 8-16 m;")


@pytest.mark.slow
def test_observe_the_line_with_its_pseudo_section(judged_line: JudgedLine) -> None:
    section = observe(judged_line.folder, "line", QCConfig()).split("Pseudo-section")[1]

    rows = section.splitlines()[2:]
    assert [row.split()[0] for row in rows] == ["17.25", "53.25", "89.25", "125.25"]
    # The curves at 89.25 and 125.25 m reach a wavelength of 10 m, none the table's others.
    assert [row.split()[1:] for row in rows[2:]] == [
        ["-", "-", "248", "-", "-"],
        ["-", "-", "283", "-", "-"],
    ]


@pytest.mark.slow
def test_observe_what_is_not_there(judged_line: JudgedLine) -> None:
    assert observe(judged_line.folder, "models", QCConfig()).startswith(
        "No G5 result on this run yet: invert the frozen curves first"
    )
    with pytest.raises(ValueError, match="Unknown filter"):
        observe(judged_line.folder, "curves", QCConfig(), where="everything")
    with pytest.raises(ValueError, match="page"):
        observe(judged_line.folder, "curves", QCConfig(), where="verdict:pass", page=2)


@pytest.mark.slow
def test_look_at_a_window_its_pick_and_its_limits(judged_line: JudgedLine) -> None:
    seen = look(judged_line.folder, "image", "89", vision=True)

    assert seen.text.startswith("xmid_89.25:")
    (figure,) = seen.figures
    assert figure.startswith(PNG)


@pytest.mark.slow
def test_without_vision_a_ridge_table_and_g1s_measures(judged_line: JudgedLine) -> None:
    image = look(judged_line.folder, "image", "89", vision=False)
    gather = look(judged_line.folder, "gather", "101.dat", vision=False)

    assert image.figures == gather.figures == ()
    assert image.text.startswith("xmid_89.25, its image as a ridge table over G2's coherent band")
    assert gather.text.startswith(
        "101.dat, G1's measures in its gather's stead (vision is off): 101.dat: reject; low_snr;"
    )
    with pytest.raises(ValueError, match="figures only"):
        look(judged_line.folder, "sheet", "0-150", vision=False)


@pytest.mark.slow
def test_look_at_records_and_a_sheet(judged_line: JudgedLine) -> None:
    assert look(judged_line.folder, "gather", "101.dat", vision=True).figures[0].startswith(PNG)
    assert look(judged_line.folder, "spectra", "117.dat", vision=True).figures[0].startswith(PNG)
    sheet = look(judged_line.folder, "sheet", "0-150", vision=True)
    assert sheet.text == "4 of the 4 windows from 0 to 150 m."
    with pytest.raises(ValueError, match="No record"):
        look(judged_line.folder, "gather", "999.dat", vision=True)


@pytest.mark.slow
def test_advise_reads_the_committed_lines_gates(judged_line: JudgedLine, tmp_path: Path) -> None:
    study = Study("active_test_profile", tmp_path)
    settings = Settings(input_dir=judged_line.folder, output_dir=tmp_path)

    def said(topic: str, run_folder: Path | None = judged_line.folder) -> str:
        return advise(study, topic, {}, settings, QCConfig(), run_folder)  # pyright: ignore[reportArgumentType]

    # One curve fails G3, which asks changes of its picking; the line loop asks none.
    assert said("picking").startswith("G3's changes of the picking, by window (not applied):\n")
    assert "- too_few_points on xmid 53.25 (1): picking" in said("picking")
    assert said("line_changes") == (
        "PACo's line loop: no change of the line's settings asked by the gates."
    )
    assert said("picking", None).startswith("No line committed yet")
    assert said("inversion").startswith("No model yet")


@pytest.mark.slow
def test_advise_makes_pacos_trials_once(input_dir: Path, tmp_path: Path) -> None:
    # Few trial windows and two lengths: PACo's ladder and mute trial in a few seconds.
    config = QCConfig.model_validate(
        {"coherence": {"trials": 3, "lengths": [9, 13]}, "mute": {"trials": 3}}
    )
    settings = Settings(input_dir=input_dir, output_dir=tmp_path / "output", workers=2)
    study = new_study("active_test_profile", settings)

    ladder = advise(study, "window_length", {}, settings, config)
    mute = advise(study, "mute", {}, settings, config)

    assert ladder.startswith("PACo's ladder, on trial windows spread along the line (not applied):")
    assert ladder.splitlines()[-1].startswith("It would choose windows of ")
    # No length given: the mute trial at the ladder's, as the workflow orders them.
    length = ladder.splitlines()[-1].split()[-2]
    assert mute.startswith(
        f"PACo's mute trial at {length} receivers (its ladder's), on trial windows spread along "
    )
    assert mute.splitlines()[-1].startswith("It would keep: ")
    # Kept in the study: asked again, the same, made before.
    assert advise(study, "window_length", {}, settings, config) == ladder
