"""One window's inversion: a port of PAC's adapters/inversion.py and io/inversion.py
(invert_position), working on a window folder of a PACo run."""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from disba import DispersionError
from sigpipe.algorithms.inversion.rayleigh.seismic.forward import (
    fwd_seismic_all_modes,
    fwd_seismic_phase,
)
from sigpipe.algorithms.picking.dispersion.curve import min_resolvable_wavelength
from sigpipe.base import DispersionCurves, Mode, Pipeline
from sigpipe.base.inversion import InversionResult
from sigpipe.dataio.dispersion.loading import load_dispersion_curves
from sigpipe.dataio.dispersion.plotting import plot_dispersion_image
from sigpipe.dataio.dispersion.saving import save_dispersion_curves
from sigpipe.dataio.inversion.forward import MODEL_NAMES, forward_model_all
from sigpipe.dataio.inversion.plotting import plot_density_curves, plot_posterior_marginals
from sigpipe.transformers import Invert, Plot, Save

from paco.inversion.models import InversionParameters
from paco.runs import load_image

logger = logging.getLogger(__name__)

# PAC's fixed values (adapters/inversion.py).
DZ = 0.01  # m
VP_VS_RATIO = 1.77
CURVES_FILE = "DispersionCurves_0000.csv"
SAMPLES_FILE = "SeismicInversion_Samples_0000.npz"  # PACo's own, next to PAC's files
M0 = Mode("M", 0)  # the label pick gives the fundamental mode


def build_inversion_pipeline(parameters: InversionParameters, output_folder: Path) -> Pipeline:
    """PAC's inversion pipeline: sigpipe's MCMC, then the models saved in `output_folder`."""
    return Invert(
        method="mcmc",
        n_layers=parameters.n_layers,
        thicknesses_min=tuple(layer.thickness_min for layer in parameters.thickness_layers),
        thicknesses_max=tuple(layer.thickness_max for layer in parameters.thickness_layers),
        thickness_perturbations=tuple(
            layer.thickness_perturb_std for layer in parameters.thickness_layers
        ),
        Vs_mins=tuple(layer.vs_min for layer in parameters.vs_layers),
        Vs_maxs=tuple(layer.vs_max for layer in parameters.vs_layers),
        Vs_perturbations=tuple(layer.vs_perturb_std for layer in parameters.vs_layers),
        n_iterations=parameters.n_iterations,
        n_burnin=parameters.n_burnin_iterations,
        n_chains=parameters.n_chains,
        Vp_Vs_ratio=VP_VS_RATIO,
        dz=DZ,
    ) >> Save(folder_path=output_folder, file_name="SeismicInversion_Model")


def invert_window(folder: Path, parameters: InversionParameters) -> InversionResult:
    """Invert the M0 curve saved in window folder `folder`, and write PAC's files next to it."""
    image = load_image(folder)
    curves = _m0_curve(folder)
    result: InversionResult = build_inversion_pipeline(parameters, folder).run(
        data=[curves], show_log=False
    )[0]

    (folder / "SeismicInversion_Log_0000.log").write_text(result.log)
    save_samples(result, parameters.n_chains, folder / SAMPLES_FILE)

    # The median model's M0 at the picked frequencies, and every mode it supports across the
    # image, drawn over the image (the old Streamlit app's pred_modes and full_pred_modes).
    median = result.median
    try:
        modeled_curves = DispersionCurves(
            dispersion_curves=tuple(
                fwd_seismic_phase(
                    thickness_per_layer=list(median.thicknesses),
                    Vs_per_layer=list(median.vs_s),
                    mode=curve.mode.number,
                    fs=curve.fs,
                    Vp_Vs_ratio=VP_VS_RATIO,
                )
                for curve in curves
            )
        )
    except DispersionError:
        # A layer over a slower half-space has no normal mode faster than the half-space: the
        # median of the samples can lack one where every sample had it. The figure goes without
        # (PAC would lose the whole window here).
        logger.warning("No mode of the median model at the picked frequencies in %s", folder)
        modeled_curves = None
    full_modeled_curves = fwd_seismic_all_modes(
        thickness_per_layer=list(median.thicknesses),
        Vs_per_layer=list(median.vs_s),
        fs=image.fs,
        Vp_Vs_ratio=VP_VS_RATIO,
    )
    figure = plot_dispersion_image(
        image,
        picked_curves=curves,
        modeled_curves=modeled_curves,
        full_modeled_curves=full_modeled_curves,
        lbmin=min_resolvable_wavelength(image.acquisition),
        normalize=True,
        show_errorbars=True,
    )
    Plot.savefig(path=folder / "SeismicInversion_DispersionImage_0000.png", figure=figure)
    plt.close(figure)

    forward_modeled = forward_model_all(result, curves, VP_VS_RATIO)
    for model_name in MODEL_NAMES:
        modeled = forward_modeled[model_name]
        if modeled is None:
            logger.warning(
                "Could not forward-model '%s' in %s; skipping its curves", model_name, folder
            )
            continue
        save_dispersion_curves(
            modeled, path=folder / f"SeismicInversion_DispersionCurves_0000_{model_name}.csv"
        )

    figure = plot_density_curves(result, curves, VP_VS_RATIO)
    Plot.savefig(path=folder / "SeismicInversion_DensityCurves_0000.png", figure=figure)
    plt.close(figure)

    samples = {f"Vs{i + 1} [m/s]": result.samples[f"vs{i + 1}"] for i in range(result.n_layers)}
    samples |= {
        f"H{i + 1} [m]": result.samples[f"thick{i + 1}"] for i in range(result.n_layers - 1)
    }
    try:
        figure = plot_posterior_marginals(samples)
        Plot.savefig(path=folder / "SeismicInversion_Marginals_0000.png", figure=figure)
        plt.close(figure)
    except Exception:  # a figure must not lose the inversion, as in PAC
        logger.exception("Could not plot the posterior marginals in %s", folder)

    return result


def save_samples(result: InversionResult, n_chains: int, path: Path) -> None:
    """The posterior samples, chain after chain as sigpipe concatenates them, with each
    sample's misfit: what G5 judges convergence and the prior's bounds on. PAC does not keep
    them."""
    arrays: dict[str, Any] = {name: np.asarray(values) for name, values in result.samples.items()}
    np.savez_compressed(
        path, n_chains=np.array(n_chains), misfits=np.asarray(result.misfits), **arrays
    )


def load_samples(path: Path) -> tuple[dict[str, np.ndarray], int]:
    """The samples `save_samples` wrote, by parameter (vs1, ..., thick1, ...), and the number
    of chains they come from."""
    with np.load(path) as saved:
        n_chains = int(saved["n_chains"])
        samples = {name: saved[name] for name in saved.files if name not in ("n_chains", "misfits")}
    return samples, n_chains


def _m0_curve(folder: Path) -> DispersionCurves:
    path = folder / CURVES_FILE
    saved = load_dispersion_curves([path])[0] if path.exists() else ()
    m0 = tuple(curve for curve in saved if curve.mode == M0)
    if not m0:
        raise ValueError(f"No M0 curve in {folder.name}: pick it, or pick it again by hand.")
    return DispersionCurves(dispersion_curves=m0)
