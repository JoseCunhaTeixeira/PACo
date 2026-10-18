"""From sigpipe, src/sigpipe/masw/inversion/window.py, `build_inversion_pipeline`, `invert_window`
and `_curves`, at commit 772ff36.

Why: sigpipe's `inversion_mcmc` takes a seed, which its `Invert` transformer passes on, but
`invert_window` builds the pipeline without one, so two runs of a window draw other models. The
geophysicist agent seeds each inversion (docs/geo_plan.md §10), so that the package's replay makes
it again exactly. Copied as they are, but the seed."""

import logging
from collections.abc import Collection
from pathlib import Path

from sigpipe.base.dispersion_curve import DispersionCurves, Mode
from sigpipe.base.inversion import InversionResult
from sigpipe.base.pipeline import Pipeline
from sigpipe.dataio.dispersion.loading import load_dispersion_curves
from sigpipe.dataio.dispersion.saving import save_dispersion_curves
from sigpipe.dataio.inversion.forward import MODEL_NAMES, forward_model_all
from sigpipe.masw.inversion import InversionParameters
from sigpipe.masw.inversion.window import (
    DZ,
    M0,
    PARAMETERS_FILE,
    SAMPLES_FILE,
    SPREAD_FILE,
    VP_VS_RATIO,
    VS_SPREAD_FILE,
    draw_figures,
    load_parameters,
    save_parameters,
    save_samples,
    save_spread,
    save_vs_spread,
    vs_spread,
)
from sigpipe.masw.picks import CURVES_FILE
from sigpipe.transformers import Invert, Save

logger = logging.getLogger(__name__)


def build_inversion_pipeline(
    parameters: InversionParameters,
    output_folder: Path,
    chain_jobs: int = 1,
    seed: int | None = None,
) -> Pipeline:
    """The inversion pipeline: sigpipe's MCMC, its chains in `chain_jobs` processes, seeded with
    `seed` (None: unseeded), then the models saved in `output_folder`."""
    mcmc = Invert(
        method="mcmc",
        Vp_Vs_ratio=VP_VS_RATIO,
        dz=DZ,
        chain_jobs=chain_jobs,
        seed=seed,
        **dict(parameters),
    )
    return mcmc >> Save(folder_path=output_folder, file_name="SeismicInversion_Model")


def invert_window(
    folder: Path,
    parameters: InversionParameters,
    modes: Collection[Mode] = (M0,),
    chain_jobs: int = 1,
    output_folder: Path | None = None,
    seed: int | None = None,
) -> InversionResult:
    """Invert the curves of `modes` saved in window folder `folder` (M0 by default), the chains
    in `chain_jobs` processes seeded with `seed`, and write PAC's files next to them, or in
    `output_folder` (a staging folder: see sigpipe.masw.runs.stopping)."""
    out = output_folder or folder
    curves = _curves(folder, modes)
    result: InversionResult = build_inversion_pipeline(parameters, out, chain_jobs, seed).run(
        data=[curves], show_log=False
    )[0]

    (out / "SeismicInversion_Log_0000.log").write_text(result.log)
    save_parameters(parameters, result, out / PARAMETERS_FILE)
    save_samples(result, out / SAMPLES_FILE)
    save_spread(result, curves, out / SPREAD_FILE)
    ran = load_parameters(out / PARAMETERS_FILE).parameters
    # As sigpipe's: an inversion's result holds its profiles.
    spread = vs_spread(result.profiles, ran.bottom)  # pyright: ignore[reportArgumentType]
    save_vs_spread(spread, out / VS_SPREAD_FILE)

    # Each model's curves at the picked frequencies, saved beside it; the figures drawn from the
    # files saved.
    forward_modeled = forward_model_all(result, curves, VP_VS_RATIO)
    for model_name in MODEL_NAMES:
        modeled = forward_modeled[model_name]
        if modeled is None:
            logger.warning(
                "Could not forward-model '%s' in %s; skipping its curves", model_name, folder
            )
            continue
        save_dispersion_curves(
            modeled, path=out / f"SeismicInversion_DispersionCurves_0000_{model_name}.csv"
        )
    draw_figures(folder, output_folder)
    return result


def _curves(folder: Path, modes: Collection[Mode]) -> DispersionCurves:
    path = folder / CURVES_FILE
    saved = load_dispersion_curves([path])[0] if path.exists() else ()
    chosen = tuple(curve for curve in saved if curve.mode in modes)
    if not chosen:
        labels = " or ".join(sorted(mode.label for mode in modes))
        raise ValueError(f"No {labels} curve in {folder.name}: pick it, or pick it again by hand.")
    return DispersionCurves(dispersion_curves=chosen)
