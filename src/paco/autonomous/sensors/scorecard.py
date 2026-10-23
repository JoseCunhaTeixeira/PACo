"""Scorecards: what an experiment's trial windows, or the committed line,
give, for the model to compare variants; a variant against the current best; and which variants
another one beats on every count.

The pick's uncertainty is a function of its wavelength and of the window's receivers alone
(sigpipe's Lorentzian, capped at 40 % of the velocity): precision grows with the window's length and
falls as curves reach deeper, whatever the image. It is shown, not counted: variants are compared on
their images first (G2's coherent columns, lateral coherence and background off the ridge, G3's
share of each pick on its ridge), then on their coverage and depth."""

from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from paco.autonomous.sensors.checks import failing_runs, median_known, window_checks
from paco.qc.models import GateResult

# The share of the best's depth a variant may lose: more, it is shortened.
DEPTH_LOSS = 0.2


@dataclass(frozen=True, slots=True)
class Scorecard:
    """What a set of windows gives. Medians over the covered curves, the images' over every window
    judged; None where none is known."""

    windows: tuple[float, ...]  # the windows judged, by xmid
    # Those whose curve counts: passing G3 (trial windows), or accepted (the committed line).
    covered: tuple[float, ...]
    longest_gap: tuple[float, ...]  # the longest run of consecutive windows not covered
    depth_m: float | None  # the median depth informed, half the longest wavelength
    depth_p10_m: float | None  # its 10th percentile
    shallowest_m: float | None  # the median shortest wavelength
    span: float | None  # the median longest wavelength over the shortest
    precision: float | None  # the median of the curves' median relative uncertainty
    near_field: float | None  # the share of the curves whose nearest shot is in their near field
    beyond: float | None  # the median share of points beyond the window's reach
    aliased: float | None  # the median share in the aliasing zone
    outliers: int | None  # G4's outliers among the curves, on the line
    misfit: float | None  # G4's median misfit to the neighbours, on the line
    cpu_s: float | None  # what it cost
    # The images: each window's mean of the measures below, the background as 1 minus it (1 at
    # best); their median, and their 10th percentile for the weakest windows.
    image: float | None = None
    image_p10: float | None = None
    coherent: float | None = None  # G2: the share of the image's columns holding a coherent peak
    lateral: float | None = None  # G2: the lateral coherence of the window's traces
    on_ridge: float | None = None  # G3: the share of each pick on its image's ridge (on_data)
    background: float | None = None  # G2: the level off the ridge, over its peak (0 at best)
    stripes: float | None = None  # G2: that level's jumps between columns (reported)

    @property
    def coverage(self) -> float:
        """The share of the windows covered."""
        return len(self.covered) / len(self.windows) if self.windows else 0.0


@dataclass(frozen=True, slots=True)
class Against:
    """A variant against the current best, on the windows both judged."""

    gained: tuple[float, ...]  # windows the variant covers and the best does not
    lost: tuple[float, ...]  # windows the best covers and the variant does not
    # It gains windows by shortening its curves: its median longest wavelength
    # falls by more than the line's rules allow.
    shortened: bool


def scorecard(
    g3: Mapping[float, GateResult | None],
    length_m: float,
    *,
    covered: Collection[float] | None = None,
    g4: Iterable[GateResult] = (),
    cpu_s: float | None = None,
    shots: bool = True,
    g2: Mapping[float, GateResult | None] | None = None,
) -> Scorecard:
    """The scorecard of the windows `g3` holds, by xmid: each one's G3 result, None when it gave
    none (its processing failed). `length_m`: the windows' length (sigpipe's window_length).
    `covered`: the windows whose curve counts, by default those passing G3. `g4`: G4's results
    for the curves, on the committed line. Without `shots` (a passive line), no near field. `g2`:
    G2's results on the windows' images, by xmid, for their score."""
    if covered is None:
        covered = [
            xmid for xmid, result in g3.items() if result is not None and result.verdict == "pass"
        ]
    kept = sorted(xmid for xmid in covered if xmid in g3)
    checks = [
        check
        for xmid in kept
        if (result := g3[xmid]) is not None
        and (check := window_checks(result, length_m, shots)) is not None
    ]
    depths = [check.depth_m for check in checks]
    runs = failing_runs({xmid: xmid in kept for xmid in g3})
    longest_gap = max((run.xmids for run in runs), key=len, default=())
    curves_g4 = [result for result in g4 if result.unit != "line"]
    misfits = [
        metric.value
        for result in curves_g4
        for metric in result.metrics
        if metric.name == "neighbour_misfit" and metric.value is not None
    ]
    images = g2 or {}
    measured = {
        xmid: (
            measure(images.get(xmid), "coherent_columns"),
            measure(images.get(xmid), "lateral_coherence"),
            measure(g3[xmid], "on_data"),
            measure(images.get(xmid), "background"),
        )
        for xmid in g3
    }
    scores = [
        score for xmid in g3 if (score := window_image(images.get(xmid), g3[xmid])) is not None
    ]
    return Scorecard(
        windows=tuple(sorted(g3)),
        covered=tuple(kept),
        longest_gap=longest_gap,
        depth_m=median_known(depths),
        depth_p10_m=float(np.percentile(depths, 10)) if depths else None,
        shallowest_m=median_known(check.shortest_m for check in checks),
        span=median_known(check.span for check in checks),
        precision=median_known(check.uncertainty for check in checks),
        near_field=sum(1 for check in checks if check.in_near_field) / len(checks)
        if checks and shots
        else None,
        beyond=median_known(check.beyond for check in checks),
        aliased=median_known(check.aliased for check in checks),
        outliers=sum(
            1 for result in curves_g4 if any(flag.name == "outlier" for flag in result.flags)
        )
        if curves_g4
        else None,
        misfit=median_known(misfits),
        cpu_s=cpu_s,
        image=median_known(scores),
        image_p10=float(np.percentile(scores, 10)) if scores else None,
        coherent=median_known(values[0] for values in measured.values()),
        lateral=median_known(values[1] for values in measured.values()),
        on_ridge=median_known(values[2] for values in measured.values()),
        background=median_known(values[3] for values in measured.values()),
        stripes=median_known(measure(images.get(xmid), "stripes") for xmid in g3),
    )


def window_image(g2: GateResult | None, g3: GateResult | None) -> float | None:
    """A window's image score: the mean of G2's share of coherent columns and lateral coherence,
    G3's share of the pick on the ridge, and one less G2's background; None when none is known."""
    level = measure(g2, "background")
    return _mean(
        (
            measure(g2, "coherent_columns"),
            measure(g2, "lateral_coherence"),
            measure(g3, "on_data"),
            1 - level if level is not None else None,
        )
    )


def measure(result: GateResult | None, name: str) -> float | None:
    """The value of `result`'s measure `name`, when it has one."""
    if result is None:
        return None
    return next((one.value for one in result.metrics if one.name == name), None)


def _mean(values: Iterable[float | None]) -> float | None:
    """The mean of the known `values`; None when none is."""
    known = [value for value in values if value is not None]
    return sum(known) / len(known) if known else None


def against(variant: Scorecard, best: Scorecard, loss: float = DEPTH_LOSS) -> Against:
    """`variant` against the current `best`, on the windows both judged; `loss`: the share of
    the best's depth a variant may lose before it is shortened."""
    common = set(variant.windows) & set(best.windows)
    ours, theirs = set(variant.covered) & common, set(best.covered) & common
    gained, lost = tuple(sorted(ours - theirs)), tuple(sorted(theirs - ours))
    shorter = (
        variant.depth_m is not None
        and best.depth_m is not None
        and variant.depth_m < (1 - loss) * best.depth_m
    )
    return Against(gained=gained, lost=lost, shortened=len(gained) > len(lost) and shorter)


def dominated(cards: Mapping[str, Scorecard]) -> dict[str, str]:
    """The variants of `cards` another one dominates, each with the first that does: at least as
    good on its images, its coverage and its depth, and better on one of them. An unknown image
    score or depth is the worst."""
    scores = {name: _scores(card) for name, card in cards.items()}
    beaten: dict[str, str] = {}
    for name, score in scores.items():
        for other, better in scores.items():
            if other != name and _dominates(better, score):
                beaten[name] = other
                break
    return beaten


def _scores(card: Scorecard) -> tuple[float, float, float]:
    """The images, the coverage and the depth, each larger when better."""
    return (
        card.image if card.image is not None else float("-inf"),
        card.coverage,
        card.depth_m if card.depth_m is not None else float("-inf"),
    )


def _dominates(better: tuple[float, ...], worse: tuple[float, ...]) -> bool:
    return all(b >= w for b, w in zip(better, worse, strict=True)) and better != worse
