"""survey: active_test_profile's geometry, the limits it sets, and its raw records as
G1 measures them, made once a study; look at those records before the line's first commit, a mute
and velocities drawn on a gather."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from sigpipe.masw.runs import RunError

from paco.autonomous.tools.look import Mute, look_survey
from paco.autonomous.tools.survey import Survey, describe_survey, survey
from paco.autonomous.tools.trials import Study, new_study
from paco.qc.config import QCConfig
from paco.settings import Settings

PNG = b"\x89PNG"


@dataclass(frozen=True)
class Surveyed:
    settings: Settings
    study: Study
    found: Survey


@pytest.fixture(scope="module")
def surveyed(input_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Surveyed]:
    """A study of active_test_profile, surveyed (a second or two)."""
    root = tmp_path_factory.mktemp("survey")
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(Settings.model_config, "env_file", None)
        patch.chdir(root)
        settings = Settings(input_dir=input_dir, output_dir=root / "output", workers=2)
        study = new_study("active_test_profile", settings)
        yield Surveyed(settings, study, survey(study, settings, QCConfig()))


@pytest.mark.slow
def test_the_lines_geometry_and_the_limits_it_sets(surveyed: Surveyed) -> None:
    found = surveyed.found

    assert (found.receivers, found.spacing_m, found.line_m) == (96, 1.5, 142.5)
    # The 25 shots: one 0.75 m before the line, one 0.75 m after, the others on it, 6 m apart,
    # each file stating a trigger of 20 ms.
    shots = [
        (shot.record, shot.x_m, shot.side, shot.offset_m, shot.trigger_s) for shot in found.shots
    ]
    assert len(shots) == 25
    assert shots[:2] == [
        ("101.dat", -0.75, "before the line", 0.75, 0.02),
        ("105.dat", 5.25, "on the line", 0.0, 0.02),
    ]
    assert shots[-1] == ("197.dat", 143.25, "after the line", 0.75, 0.02)
    assert (found.sampling_hz, found.nyquist_hz) == (2000, 1000)
    assert found.frequency_step_hz == pytest.approx(1 / 3, abs=1e-3)
    # Twice the spacing; windows of 5 receivers to half the line.
    assert (found.shortest_wavelength_m, found.lengths) == (3.0, (5, 48))


@pytest.mark.slow
def test_the_raw_records_as_g1_measures_them(surveyed: Surveyed) -> None:
    found = surveyed.found

    assert found.reach_m == pytest.approx(66.64, abs=0.01)
    assert found.cone_m_s == pytest.approx((125.3, 202.5, 332.2), abs=0.1)
    assert found.cone_traces == 1086
    # Two records under G1's SNR, left out of every window.
    assert {unit for unit, result in found.g1.items() if result.verdict != "pass"} == {
        "101.dat",
        "129.dat",
    }
    assert found.g1["line"].verdict == "pass"


@pytest.mark.slow
def test_the_survey_as_the_model_reads_it_made_once(surveyed: Surveyed) -> None:
    lines = describe_survey(surveyed.found).splitlines()

    # Many shots, and many records, summed up.
    assert lines[0] == (
        "active_test_profile: 96 receivers 1.5 m apart, 142.5 m long; 25 shots: 1 before the line (at "
        "-0.75 m), 23 on the line (from 5.25 to 137.25 m every 6 m), 1 after the line (at 143.25 m), "
        "their files' trigger 0.02 s."
    )
    assert lines[2].endswith(
        "windows of 5 to 48 receivers, N_r - N + 1 of them at step 1 (92 at 5, 71 at 26, 49 at 48)."
    )
    assert lines[4].startswith("Surface waves' cone: 125 to 332 m/s, median 202.5")
    assert lines[-4].startswith("- 101.dat: reject; low_snr; ")
    assert lines[-2:] == [
        "- the line: pass; off_decay_receivers; off_decay_receivers 1, spectral_receivers 0",
        "- 23 records: pass, no flag",
    ]
    assert survey(surveyed.study, surveyed.settings, QCConfig()) == surveyed.found


@pytest.mark.slow
def test_regular_shots_are_said_with_their_step(surveyed: Surveyed) -> None:
    found = surveyed.found
    regular = tuple(
        shot.model_copy(update={"record": f"{100 + n}.dat", "x_m": 0.75 + 1.5 * n})
        for n, shot in enumerate([found.shots[1]] * 95)
    )

    said = describe_survey(found.model_copy(update={"shots": regular})).splitlines()[0]

    assert said.endswith(
        "95 shots: 95 on the line (from 0.75 to 141.75 m every 1.5 m), their files' trigger 0.02 s."
    )


@pytest.mark.slow
def test_look_at_the_raw_records_before_the_first_commit(surveyed: Surveyed) -> None:
    study = surveyed.study
    mute = Mute(vmin=80, vmax=1500)

    gather = look_survey(study, "gather", "105", vision=True)
    spectra = look_survey(study, "spectra", "105.dat", vision=True)
    drawn = look_survey(study, "gather", "105.dat", True, mute, velocities=[150.0, 300.0])
    blind = look_survey(study, "gather", "101.dat", False, mute)

    # A record named with or without its extension; its figures as the survey drew them.
    assert (gather.text, spectra.text) == ("105.dat: its raw gather.", "105.dat: its raw spectra.")
    assert all(seen.figures[0].startswith(PNG) for seen in (gather, spectra, drawn))
    # The mute's tail PACo's when it gives none.
    assert drawn.text == (
        "105.dat: its raw gather, the mute's window in red, from x/1500 to x/80 + 0.05 s; lines from "
        "the shot at 150 m/s, 300 m/s in blue, dashed."
    )
    assert blind.figures == ()
    assert blind.text.startswith(
        "101.dat, G1's measures in its gather's stead (vision is off): 101.dat: reject; low_snr;"
    )
    with pytest.raises(ValueError, match="drawn on a gather, not on spectra"):
        look_survey(study, "spectra", "105.dat", True, mute)
    with pytest.raises(ValueError, match="Before the first commit, look shows the survey's raw"):
        look_survey(study, "image", "71", vision=True)
    with pytest.raises(ValueError, match="No record '999' in the survey: 101, 105"):
        look_survey(study, "gather", "999", vision=True)


def test_look_before_the_survey_is_refused(tmp_path: Path) -> None:
    with pytest.raises(RunError, match="No survey yet in this study: call survey first"):
        look_survey(Study("active_test_profile", tmp_path), "gather", "105.dat", vision=True)
