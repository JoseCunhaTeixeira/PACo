"""survey (docs/autonomous_plan.md §4.1): active_test_profile's geometry, the limits it sets, and its raw records as
G1 measures them, made once a study."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from paco.autonomous.tools.survey import Survey, describe_survey, survey
from paco.autonomous.tools.trials import Study, new_study
from paco.qc.config import QCConfig
from paco.settings import Settings


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
    # The shots the tests keep: one 0.75 m before the line, one 0.75 m after, the others on it,
    # each file stating a trigger of 20 ms.
    shots = [
        (shot.record, shot.x_m, shot.side, shot.offset_m, shot.trigger_s) for shot in found.shots
    ]
    assert len(shots) == 28
    assert shots[:2] == [
        ("101.dat", -0.75, "before the line", 0.75, 0.02),
        ("102.dat", 0.75, "on the line", 0.0, 0.02),
    ]
    assert shots[-1] == ("197.dat", 143.25, "after the line", 0.75, 0.02)
    assert (found.sampling_hz, found.nyquist_hz) == (2000, 1000)
    assert found.frequency_step_hz == pytest.approx(1 / 3, abs=1e-3)
    # Twice the spacing; windows of 5 receivers to half the line.
    assert (found.shortest_wavelength_m, found.lengths) == (3.0, (5, 48))


@pytest.mark.slow
def test_the_raw_records_as_g1_measures_them(surveyed: Surveyed) -> None:
    found = surveyed.found

    assert found.reach_m == pytest.approx(66.0, abs=0.01)
    assert found.cone_m_s == pytest.approx((106.8, 203.4, 762.8), abs=0.1)
    assert found.cone_traces == 1088
    # Two records under G1's SNR, left out of every window.
    assert {unit for unit, result in found.g1.items() if result.verdict != "pass"} == {
        "101.dat",
        "144.dat",
    }
    assert found.g1["line"].verdict == "pass"


@pytest.mark.slow
def test_the_survey_as_the_model_reads_it_made_once(surveyed: Surveyed) -> None:
    lines = describe_survey(surveyed.found).splitlines()

    # Many shots, and many records, summed up.
    assert lines[0] == (
        "active_test_profile: 96 receivers 1.5 m apart, 142.5 m long; 28 shots: 1 before the line (at "
        "-0.75 m), 26 on the line (from 0.75 to 141.75 m), 1 after the line (at 143.25 m), their "
        "files' trigger 0.02 s."
    )
    assert lines[2].endswith(
        "windows of 5 to 48 receivers, N_r - N + 1 of them at step 1 (92 at 5, 71 at 26, 49 at 48)."
    )
    assert lines[4].startswith("Surface waves' cone: 107 to 763 m/s, median 203")
    assert lines[-4].startswith("- 101.dat: reject; low_snr; ")
    assert lines[-2:] == [
        "- the line: pass; no flag; off_decay_receivers 0, spectral_receivers 0",
        "- 26 records: pass, no flag",
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
