"""tidy: the study's working files removed, its results kept, so that only the results stay: at the
freeze, once nothing is processed or picked again, and when its conversation ends (a conversation
is its study). Working files: the records each base preprocessed, the trials' windows, G1's settled
records, the search's scratch and thumbnails, and once the curves are frozen the survey's figures.
Results, kept: the line's run (PAC's layout, its own records as PAC's runs keep them), the frozen
curves, the journal, the decisions and picks, the experiments' and the search's scores, each
sheet's judgment, the transcripts. A study opened again after its report makes again what it needs
(a trial on try_settings); a search whose sheets were not all judged is taken back (its scores and
its sheets)."""

import shutil
from pathlib import Path

from paco.autonomous.tools import grid, survey
from paco.autonomous.tools import search as searching
from paco.autonomous.tools.trials import CURVES, G1, RECORDS, TRIALS, Study


def tidy(study: Study) -> int:
    """`study`'s working files removed: the bytes freed."""
    folder = study.folder
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


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return (
        sum(one.stat().st_size for one in path.rglob("*") if one.is_file()) if path.is_dir() else 0
    )
