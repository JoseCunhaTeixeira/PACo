"""The line's velocity section, as PAC writes it at the end of an inversion run: the smooth
median's Vs(x, z) and its spread as a figure, every model variant's section in one HDF5 file,
and the picked curves against the curves the smooth medians predict, in the run folder."""

import logging
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
from sigpipe.base.dispersion_curve import DispersionCurve, DispersionCurvesSection
from sigpipe.base.velocity_model import VelocityModelsSection
from sigpipe.dataio.dispersion.loading import load_dispersion_curves
from sigpipe.dataio.dispersion.section import plot_pseudo_section_comparison
from sigpipe.dataio.inversion.forward import MODEL_NAMES
from sigpipe.dataio.velocity_model.loading import load_velocity_models
from sigpipe.dataio.velocity_model.section import (
    plot_velocity_and_std_section,
    save_velocity_models_sections,
)
from sigpipe.transformers import Plot

from paco.inversion.window import DZ
from paco.picks import CURVES_FILE

logger = logging.getLogger(__name__)

SECTION_FIGURE = "SeismicInversion_VelocitySection_0000.png"
SECTION_FILE = "SeismicInversion_VelocitySection_0000.hdf5"
COMPARISON_FIGURE = "SeismicInversion_PseudoSectionComparison_0000_M0.png"


def save_section(run_folder: Path, units: Sequence[str]) -> Path | None:
    """PAC's section outputs over the window folders `units` (the models G5 passed): the figure
    of the smooth median (PAC's default view) and the HDF5 file of every variant. Returns the
    figure's path; None with fewer than two models, as PAC needs two positions for a section."""
    sections: dict[str, VelocityModelsSection] = {}
    for model in MODEL_NAMES:
        paths = [run_folder / unit / f"SeismicInversion_Model_0000_{model}.csv" for unit in units]
        found = [path for path in paths if path.exists()]
        if len(found) < 2:
            continue
        models = sorted(
            (load_velocity_models([path])[0][0] for path in found),
            key=lambda one: one.position.x,
        )
        sections[model] = VelocityModelsSection(velocity_models=tuple(models))
    if "smooth_median" not in sections:
        return None
    figure = plot_velocity_and_std_section(sections["smooth_median"], dz=DZ)
    path = run_folder / SECTION_FIGURE
    Plot.savefig(path=path, figure=figure)
    plt.close(figure)
    save_velocity_models_sections(sections, run_folder / SECTION_FILE, dz=DZ)
    return path


def save_comparison(run_folder: Path, units: Sequence[str]) -> Path | None:
    """PAC's pseudo-section comparison over the window folders `units`: each picked M0 against
    the M0 its smooth median predicts (saved by the inversion), along the line. None with fewer
    than two windows holding both."""
    observed: list[DispersionCurve] = []
    predicted: list[DispersionCurve] = []
    for unit in units:
        picked_path = run_folder / unit / CURVES_FILE
        modeled_path = (
            run_folder / unit / "SeismicInversion_DispersionCurves_0000_smooth_median.csv"
        )
        if not (picked_path.exists() and modeled_path.exists()):
            continue
        picked = next(
            (one for one in load_dispersion_curves([picked_path])[0] if one.mode.number == 0), None
        )
        modeled = next(
            (one for one in load_dispersion_curves([modeled_path])[0] if one.mode.number == 0),
            None,
        )
        if picked is None or modeled is None:
            continue
        observed.append(picked)
        # The forward model knows no position: the picked curve's, as PAC gives it.
        predicted.append(
            DispersionCurve(
                fs=modeled.fs,
                vs=modeled.vs,
                mode=modeled.mode,
                acquisition=picked.acquisition,
                type=modeled.type,
            )
        )
    if len(observed) < 2:
        return None
    figure = plot_pseudo_section_comparison(
        DispersionCurvesSection(dispersion_curves=tuple(observed)),
        DispersionCurvesSection(dispersion_curves=tuple(predicted)),
    )
    path = run_folder / COMPARISON_FIGURE
    Plot.savefig(path=path, figure=figure)
    plt.close(figure)
    return path
