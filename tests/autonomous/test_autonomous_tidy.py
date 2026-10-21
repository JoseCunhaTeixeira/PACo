"""tidy: the study's working files removed at the freeze and when a session ends, its results
kept; a search whose sheets were not all judged taken back."""

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from sigpipe.masw.runs.finding import IMAGE_FILE

from paco.autonomous.host.loop import Host
from paco.autonomous.sensors.scorecard import Scorecard
from paco.autonomous.tools import grid, look
from paco.autonomous.tools import search as searching
from paco.autonomous.tools.tidy import tidy
from paco.autonomous.tools.trials import (
    CURVES,
    EXPERIMENT_IMAGES,
    EXPERIMENTS,
    TRIALS,
    Experiment,
    Study,
    Tried,
)


def _study(folder: Path, finalists: tuple[str, ...]) -> Study:
    for name in ("records/a1/b", "trials/t1/xmid_10.00", "g1/s1", "grid/b0", "survey"):
        (folder / name).mkdir(parents=True)
        (folder / name / "data.bin").write_bytes(b"x" * 1000)
    searches = folder / searching.SEARCHES
    (searches / searching.IMAGES).mkdir(parents=True)
    (searches / searching.IMAGES / "g1.npz").write_bytes(b"x" * 500)
    (searches / "S1.json").write_text("{}")
    (searches / searching.JUDGED).write_text('{"round": 1}\n')
    knockout = searching.Knockout(
        round=1, sheets=(searching.Sheet(number=1, labels=("g1", "g2")),), finalists=finalists
    )
    (searches / searching.KNOCKOUT).write_text(knockout.model_dump_json())
    for name in ("journal.jsonl", "decisions.jsonl", "line.json", "experiments/E1.json"):
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        (folder / name).write_text("{}")
    return Study("active_test_profile", folder)


def test_the_working_files_go_and_the_results_stay(tmp_path: Path) -> None:
    study = _study(tmp_path / "study", finalists=("g1",))
    (study.folder / CURVES).write_text("{}")

    freed = tidy(study)

    left = sorted(
        str(path.relative_to(study.folder)) for path in study.folder.rglob("*") if path.is_file()
    )
    assert freed == 5 * 1000 + 500
    # The records, the trials, G1, the scratch, the thumbnails and, once frozen, the survey's
    # figures gone; the journal, the decisions, the line, the experiments, the search's scores,
    # its sheets and judgments kept.
    assert left == [
        "curves.json", "decisions.jsonl", "experiments/E1.json", "journal.jsonl", "line.json",
        "searches/S1.json", "searches/S1_judged.jsonl", "searches/S1_knockout.json",
    ]  # fmt: skip
    assert tidy(study) == 0


def test_each_experiments_dispersion_images_are_kept_as_thumbnails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = _study(tmp_path / "study", finalists=("g1",))
    (study.folder / TRIALS / "t1" / "xmid_10.00" / IMAGE_FILE).write_text("")
    card = Scorecard((10.0,), (10.0,), (), *([None] * 11))
    tried = Tried(
        label="current",
        changes={},
        trial="t1",
        taken=False,
        seconds=0.0,
        scorecard=card,
        against=None,
        dominated_by=None,
    )
    made = Experiment(
        id="E1",
        profile=study.profile,
        windows="spread:1",
        xmids=(10.0,),
        current={},
        tried=(tried,),
        made_at=datetime.now(UTC),
    )
    (study.folder / EXPERIMENTS / "E1.json").write_text(made.model_dump_json())
    empty = np.zeros(0, dtype=np.float32)
    thumb = grid.Thumb(10.0, np.zeros((2, 2), np.uint8), (0, 1, 0, 1), 1.0, (empty, empty), (0, 0))
    monkeypatch.setattr(grid, "thumbnail", lambda _folder, _xmid: thumb)  # pyright: ignore[reportUnknownLambdaType]

    tidy(study)

    # Its trial's windows gone, its dispersion images kept, as a sheet keeps them.
    kept = study.folder / EXPERIMENTS / EXPERIMENT_IMAGES / "t1.npz"
    assert not (study.folder / TRIALS).exists()
    assert [one.xmid for one in grid.read_thumbs(kept)] == [10.0]


def test_a_search_not_judged_to_the_end_is_taken_back(tmp_path: Path) -> None:
    study = _study(tmp_path / "study", finalists=())

    tidy(study)

    # Its sheets could not be drawn again: the next session searches again; the judgments made
    # stay, and the survey's figures before the freeze.
    assert searching.searched(study) is None and searching.knockout_of(study) is None
    assert (study.folder / searching.SEARCHES / searching.JUDGED).exists()
    assert (study.folder / "survey" / "data.bin").exists()


def test_a_session_ends_with_its_study_tidied(tmp_path: Path) -> None:
    study = _study(tmp_path / "study", finalists=("g1",))
    host = Host.__new__(Host)
    host.study = study
    empty = Host.__new__(Host)
    empty.study = None

    assert host.close() > 0 and not (study.folder / "records").exists()
    assert empty.close() == 0


def test_an_experiment_whose_images_were_removed_says_so(tmp_path: Path) -> None:
    study = _study(tmp_path / "study", finalists=("g1",))
    tidy(study)
    tried = Tried.model_construct(label="current", changes={}, trial="t1")

    with pytest.raises(ValueError, match="images were removed with the study's working files"):
        look._row(study, tried)  # pyright: ignore[reportPrivateUsage]
