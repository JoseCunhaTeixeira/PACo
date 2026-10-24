"""The automatic picker on a test window; its own tests, on synthetic images, are sigpipe's
(sigpipe.algorithms.picking.dispersion.tracking)."""

from datetime import UTC, datetime

import numpy as np
import pytest
from sigpipe.algorithms.picking.dispersion.tracking import PickingParameters, pick_modes
from sigpipe.base import DispersionImage
from sigpipe.masw.pipelines import build_image_pipeline, build_preprocessing_pipeline, record_folder
from sigpipe.masw.presets import make_preset, resolve_preset
from sigpipe.masw.profiles import Profile
from sigpipe.masw.windows import build_windows
from sigpipe.transformers import Load

from paco.qc.judging import HowPicked, how_picked
from paco.qc.models import Attempt


@pytest.fixture(scope="module")
def demo_image(
    profiles: dict[str, Profile], tmp_path_factory: pytest.TempPathFactory
) -> DispersionImage:
    """The dispersion image of active_test_profile's first 24-receiver window, as run_processing makes it
    from its nearest shot on each side, 18 and 30 m from its middle: the farther ones, up to
    126 m away, blur its ridge."""
    root = tmp_path_factory.mktemp("window")
    folder = root / "window"
    folder.mkdir()
    profile = profiles["active_test_profile"]
    preset = resolve_preset(
        make_preset("active", {"masw": {"length": 24, "distance_max": 40}}), profile
    )
    window = build_windows(profile, preset.masw)[0]
    with pytest.MonkeyPatch.context() as patch:
        # sigpipe's Pipeline.run creates a logs/ folder in the working directory.
        patch.chdir(root)
        for record in profile.records:
            records = record_folder(root / "records", record)
            records.mkdir(parents=True)
            build_preprocessing_pipeline(preset, record, profile, records).run(show_log=False)
        build_image_pipeline(preset, window, root / "records", folder).run(show_log=False)

    (image,) = Load(
        file_paths=[folder / "DispersionImage_0000.hdf5"], data_type="dispersion_image"
    ).transform()
    assert isinstance(image, DispersionImage)
    return image


@pytest.mark.slow
def test_m0_of_a_demo_window(demo_image: DispersionImage) -> None:
    (mode,) = pick_modes(demo_image)
    band = (mode.frequencies >= 20) & (mode.frequencies <= 40)
    kept = mode.kept & band

    # The Rayleigh wave's ridge, below the air wave (about 340 m/s).
    assert kept.sum() >= band.sum() / 2
    assert 150 < np.median(mode.velocities[kept]) < 250
    # Up to 60 Hz, no kept point strays from that ridge.
    ridge = mode.kept & (mode.frequencies >= 20) & (mode.frequencies <= 60)
    velocities = mode.velocities[ridge]
    assert velocities == pytest.approx(np.median(velocities), rel=0.2)


def test_a_pick_says_how_it_was_made() -> None:
    plain = PickingParameters()
    guided = PickingParameters(guide=((5.0, 300.0), (20.0, 250.0)))
    retried = Attempt(
        unit="xmid_1.00",
        stage="picking",
        attempt=2,
        parameters=guided.model_dump(mode="json"),
        triggered_by="G4:outlier",
        started_at=datetime(2026, 10, 24, tzinfo=UTC),
        status="succeeded",
        how="neighbours",
    )

    assert how_picked(plain, "initial", None) == HowPicked("tracker")
    # A guide: the neighbours' median curve for G4's retry, else points.
    assert how_picked(guided, "G4:outlier", None) == HowPicked("neighbours")
    assert how_picked(guided, "backtrack", None) == HowPicked("guide")
    # A check's retry keeps the guide of the attempt before: the neighbours' still.
    assert how_picked(guided, "G3:narrow_span", retried) == HowPicked("neighbours")
    assert how_picked(plain, "G3:narrow_span", retried) == HowPicked("tracker")
