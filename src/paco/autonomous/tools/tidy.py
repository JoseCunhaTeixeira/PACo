"""tidy: the study's working files removed, its results kept, so that only the results stay: at the
freeze, once nothing is processed or picked again, and when its conversation ends (a conversation
is its study). Working files: the records each base preprocessed, the trials' windows, G1's settled
records, the search's scratch and thumbnails, and once the curves are frozen the survey's figures.
Results, kept: the line's run (PAC's layout, its own records as PAC's runs keep them), the frozen
curves, the journal, the decisions and picks, the experiments' and the search's scores, each
experiment's dispersion images as thumbnails (each variant's on its windows), each sheet's
judgment, the transcripts. A study opened again after its report makes again what it needs (a
trial on try_settings); a search whose sheets were not all judged is taken back (its scores and
its sheets)."""

import shutil
from pathlib import Path

from pydantic import ValidationError
from sigpipe.masw.runs import window_folders, xmid_of
from sigpipe.masw.runs.finding import IMAGE_FILE

from paco.autonomous.tools import grid, survey
from paco.autonomous.tools import search as searching
from paco.autonomous.tools.trials import (
    CURVES,
    EXPERIMENT_IMAGES,
    EXPERIMENTS,
    G1,
    RECORDS,
    TRIALS,
    Experiment,
    Study,
)


def tidy(study: Study) -> int:
    """`study`'s working files removed, its experiments' images kept first: the bytes freed."""
    folder = study.folder
    keep_images(study)
    gone = [folder / RECORDS, folder / TRIALS, folder / G1, folder / grid.GRID]
    gone.append(folder / searching.SEARCHES / searching.IMAGES)
    if (folder / CURVES).exists():
        gone.append(folder / survey.SURVEY)
    knockout = searching.knockout_of(study)
    if knockout is not None and not knockout.finalists:
        # Its sheets cannot be drawn again: nothing goes on with them.
        gone += [folder / searching.SEARCHES / f"{searching.SEARCH}.json"]
        gone += [folder / searching.SEARCHES / searching.KNOCKOUT]
    freed = 0
    for path in gone:
        freed += _size(path)
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
    return freed


def keep_images(study: Study) -> None:
    """Each experiment's variants' dispersion images, while their trials' windows are there, kept
    as a sheet keeps them (grid.write_thumbs): <study>/experiments/images/<trial>.npz, a trial's
    windows in order along the line; one kept before, or without windows, left as it is."""
    kept = study.folder / EXPERIMENTS / EXPERIMENT_IMAGES
    for path in sorted((study.folder / EXPERIMENTS).glob("E*.json")):
        try:
            experiment = Experiment.model_validate_json(path.read_text())
        except ValidationError:
            continue
        for tried in experiment.tried:
            source, target = study.folder / TRIALS / tried.trial, kept / f"{tried.trial}.npz"
            if target.exists() or not source.is_dir():
                continue
            thumbs = [
                grid.thumbnail(source / name, xmid_of(name))
                for name in window_folders(source)
                if (source / name / IMAGE_FILE).exists()
            ]
            if thumbs:
                kept.mkdir(parents=True, exist_ok=True)
                grid.write_thumbs(target, thumbs)


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return (
        sum(one.stat().st_size for one in path.rglob("*") if one.is_file()) if path.is_dir() else 0
    )
