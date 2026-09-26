"""Attempts kept inside one run: the final result stays at the top of a window's folder, in PAC's
layout, and the results of earlier attempts move to xmid_<x>/attempts/<n>_<stage>/ when a stage
is done again. Going back to a stage invalidates everything downstream (rule 3), for that
window only."""

import shutil
from pathlib import Path

from paco.qc.models import STAGES, Stage, stage_index

ATTEMPTS_FOLDER = "attempts"

# What each stage writes in a window's folder; the picking redraws the image's figure.
STAGE_FILES: dict[Stage, tuple[str, ...]] = {
    "preprocessing": (),  # per record, in records/<record>/: nothing in a window's folder
    "phase_shift": (
        "DispersionImage_0000.hdf5",
        "DispersionImage_0000.png",
        "Stream_*.png",
        "Stream_*.hdf5",
        "error.log",
    ),
    "picking": ("DispersionCurves_*.csv", "quality.json"),
    "inversion": ("SeismicInversion_*", "inversion_error.log"),
    "petro_inversion": ("PetroInversion_*",),
}


def downstream(stage: Stage) -> tuple[Stage, ...]:
    """`stage` and every stage after it that uses its results: the petrophysical inversion reads
    the picks, not the seismic inversion's models."""
    later = STAGES[stage_index(stage) :]
    if stage == "inversion":
        return tuple(one for one in later if one != "petro_inversion")
    return later


def invalidate(window_folder: Path, stage: Stage, attempt: int) -> Path:
    """Move the window's results of `stage` and the stages after it into
    attempts/<attempt>_<stage>/, and return that folder; the window is then ready for the stage
    to run again."""
    archive = window_folder / ATTEMPTS_FOLDER / f"{attempt}_{stage}"
    archive.mkdir(parents=True)
    for later in downstream(stage):
        for pattern in STAGE_FILES[later]:
            for path in sorted(window_folder.glob(pattern)):
                shutil.move(path, archive / path.name)
    return archive


def archived_attempts(window_folder: Path) -> tuple[Path, ...]:
    folder = window_folder / ATTEMPTS_FOLDER
    return tuple(sorted(folder.iterdir())) if folder.exists() else ()


def invalidate_record(record_folder: Path, attempt: int) -> Path:
    """Move a record's preprocessed stream and figure into its attempts/<attempt>_preprocessing/,
    before the record is preprocessed again; the windows that use it go on reading the new one."""
    archive = record_folder / ATTEMPTS_FOLDER / f"{attempt}_preprocessing"
    archive.mkdir(parents=True)
    for pattern in ("Stream_*", "error.log"):
        for path in sorted(record_folder.glob(pattern)):
            shutil.move(path, archive / path.name)
    return archive
