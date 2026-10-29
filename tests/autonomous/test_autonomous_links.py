"""A trial folder's records: the study's, given as a link that reads like a folder of its own (a
symbolic link; on Windows, a directory junction, which any user may make), known as linked, and
removed with the trial folder without what it links to."""

import shutil
from pathlib import Path

from paco.autonomous.tools.trials import link_records, linked


def test_a_trial_folder_reads_the_studys_records_through_its_link(tmp_path: Path) -> None:
    store = tmp_path / "study" / "records" / "base" / "records"
    (store / "101.dat").mkdir(parents=True)
    (store / "101.dat" / "preprocessed.h5").write_text("traces")
    trial = tmp_path / "study" / "trials" / "T1"
    trial.mkdir(parents=True)

    link_records(store, trial / "records")

    assert (trial / "records" / "101.dat" / "preprocessed.h5").read_text() == "traces"
    assert linked(trial / "records")
    assert not linked(store)
    # The trial folder removed as the tidy removes it: what it linked to stays.
    shutil.rmtree(trial)
    assert (store / "101.dat" / "preprocessed.h5").read_text() == "traces"
