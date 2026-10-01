"""What a scenario finds before its conversation: runs already made, as a person leaves them in
PAC's pages or the assistant in an earlier conversation, in the scenario's outputs."""

import numpy as np
from sigpipe.algorithms.picking.dispersion.tracking import PickingParameters, pick_modes
from sigpipe.base.dispersion_curve import DispersionCurve
from sigpipe.masw.picks import save_pick
from sigpipe.masw.runs import find_run, load_image, load_manifest, run_processing
from sigpipe.masw.runs.origin import M0, mark_edited

from paco import qc
from paco.settings import Settings

# Four 24-receiver windows along the active demo line: seconds to process.
SMALL_WINDOWS = {"masw": {"length": 24, "step": 24}}
# The window whose curve hand_curve_run picks by hand.
HAND_WINDOW = "xmid_8.88"


def hand_run(settings: Settings) -> None:
    """active_p1 processed in PAC's pages (no check), its M0 picked by hand in every window
    where the picker finds one: a person's work, verified by them."""
    run_id = run_processing("active_p1", "active", SMALL_WINDOWS, settings).run_id
    run_folder = find_run(run_id, settings)
    for window in load_manifest(run_id, settings).windows:
        folder = run_folder / window.folder
        image = load_image(folder)
        modes = pick_modes(image, PickingParameters())
        if modes and modes[0].curve is not None:
            save_pick(folder, image, modes[0].curve)
            mark_edited(folder, M0)


def imaged_run(settings: Settings) -> str:
    """active_p1 processed by the assistant, as an earlier conversation leaves it: its images,
    no curve yet. Its run_id."""
    config = qc.load_qc_config(settings.qc_config)
    return qc.process_line("active_p1", SMALL_WINDOWS, settings, config).run_id


def picked_run(settings: Settings) -> str:
    """active_p1 processed and picked by the assistant, as an earlier conversation leaves it.
    Its run_id."""
    run_id = imaged_run(settings)
    qc.pick_line(run_id, settings)
    return run_id


def hand_curve_run(settings: Settings) -> None:
    """active_p1 processed and picked by the assistant, then xmid 8.88's M0 picked again by hand
    in PAC."""
    run_id = picked_run(settings)
    folder = find_run(run_id, settings) / HAND_WINDOW
    image = load_image(folder)
    fs = np.linspace(10.0, 40.0, 16)
    curve = DispersionCurve(fs=fs, vs=350.0 - 4.0 * fs, mode=M0, acquisition=image.acquisition)
    save_pick(folder, image, curve)
    mark_edited(folder, M0)
