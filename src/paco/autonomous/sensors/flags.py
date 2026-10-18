"""The gates' flags as the model reads them (docs/autonomous_plan.md §5.2): each flag's class, derived from
its action (paco.qc.models), so that no threshold of the package's own is needed.

- blocking: the window cannot be accepted at that stage until a later attempt clears it;
- fixable: a hint ("G3 suggests picking {...}") the model may follow, or try something else;
- note: acceptable, with a one-line reason when the model accepts a window that carries it;
- automatic: G1's exclusions of traces and records, which G1 applies itself and reports, since no
  setting fixes them (§2.5).

A flag's action can depend on where its gate runs: G3's prominence is a Reject on a run's picks and
an Override in trials, where a mute can still be tried. The class is therefore read from each flag;
FLAGS lists the classes each flag of G1 to G6 can take."""

from typing import Literal

from paco.qc.models import Flag

type FlagClass = Literal["blocking", "fixable", "note", "automatic"]

BLOCKING: frozenset[FlagClass] = frozenset({"blocking"})
FIXABLE: frozenset[FlagClass] = frozenset({"fixable"})
NOTE: frozenset[FlagClass] = frozenset({"note"})
AUTOMATIC: frozenset[FlagClass] = frozenset({"automatic"})
# Where the gate's context decides: what can still be tried, and what was given.
FIXABLE_OR_NOTE: frozenset[FlagClass] = frozenset({"fixable", "note"})
FIXABLE_OR_BLOCKING: frozenset[FlagClass] = frozenset({"fixable", "blocking"})

# Every flag G1 to G6 can raise, by gate and name, with the classes it can take.
FLAGS: dict[tuple[str, str], frozenset[FlagClass]] = {
    # G1, each record
    ("G1", "dead_traces"): AUTOMATIC,
    ("G1", "clipped_traces"): AUTOMATIC,
    ("G1", "nan_traces"): AUTOMATIC,
    ("G1", "record_too_short"): AUTOMATIC,
    ("G1", "no_usable_band"): AUTOMATIC,
    ("G1", "low_snr"): AUTOMATIC,
    ("G1", "low_coherence"): AUTOMATIC,
    ("G1", "inconsistent_trigger"): AUTOMATIC,
    ("G1", "shifted_trigger"): FIXABLE_OR_NOTE,  # a note when no trigger can correct it
    ("G1", "spectral_receivers"): NOTE,
    ("G1", "off_decay_receivers"): AUTOMATIC,
    ("G1", "fast_arrivals"): FIXABLE,
    # G2, each dispersion image
    ("G2", "noisy_correlations"): FIXABLE_OR_BLOCKING,  # fixable on a passive line's FK window
    ("G2", "bad_correlations"): NOTE,
    ("G2", "few_segments_kept"): FIXABLE_OR_BLOCKING,
    ("G2", "no_coherent_energy"): FIXABLE_OR_NOTE,  # a note when no mute is left to try
    ("G2", "weak_coherence"): FIXABLE_OR_NOTE,
    ("G2", "ridge_at_vmin"): FIXABLE_OR_NOTE,  # a note at the velocity floor
    ("G2", "ridge_at_vmax"): FIXABLE,
    ("G2", "band_at_fmin"): NOTE,
    ("G2", "band_at_fmax"): NOTE,
    ("G2", "aliasing"): FIXABLE,
    ("G2", "competing_ridges"): NOTE,
    ("G2", "narrower_than_usable"): NOTE,
    # G3, each curve
    ("G3", "no_ridge"): FIXABLE,
    ("G3", "sharpness"): BLOCKING,
    ("G3", "prominence"): FIXABLE_OR_BLOCKING,  # blocking on a run's picks: no mute to try there
    ("G3", "on_data"): FIXABLE,
    ("G3", "constant_wavelength"): FIXABLE_OR_NOTE,  # a note when no cut removes it
    ("G3", "aliasing_zone"): NOTE,
    ("G3", "beyond_reach"): NOTE,
    ("G3", "too_few_points"): FIXABLE,
    ("G3", "narrow_span"): FIXABLE,
    ("G3", "mode_jump"): FIXABLE,
    ("G3", "air_wave"): BLOCKING,
    ("G3", "inverse_dispersion"): NOTE,
    ("G3", "near_field"): NOTE,
    # G4, the curves along the line
    ("G4", "outlier"): FIXABLE,
    ("G4", "shared_change"): NOTE,
    ("G4", "no_neighbours"): NOTE,
    ("G4", "gaps"): NOTE,
    ("G4", "uneven_depth"): NOTE,
    ("G4", "inverse_trend"): NOTE,
    # G5, each model
    ("G5", "no_mode"): BLOCKING,  # its suggested pick is for the user: the model is rejected
    ("G5", "few_samples"): NOTE,
    ("G5", "multimodal"): NOTE,
    ("G5", "not_converged"): FIXABLE,
    ("G5", "at_bound"): FIXABLE,
    ("G5", "uninformed"): BLOCKING,
    ("G5", "smoothing_misfit"): NOTE,
    ("G5", "underfit"): FIXABLE_OR_BLOCKING,  # blocking with free layers, or at the most layers
    ("G5", "alike_layers"): FIXABLE,
    ("G5", "acceptance"): NOTE,
    ("G5", "thin_layer"): FIXABLE,
    ("G5", "deep_interface"): NOTE,
    ("G5", "most_layers"): NOTE,
    ("G5", "too_deep"): FIXABLE_OR_NOTE,  # a note when the layers leave no room to shrink
    ("G5", "strong_inversion"): NOTE,
    ("G5", "implausible_vs"): NOTE,
    # G6, the models along the line
    ("G6", "model_change"): NOTE,
    ("G6", "non_unique"): NOTE,
    ("G6", "no_neighbours"): NOTE,
    ("G6", "gaps"): NOTE,
    ("G6", "uneven_depth_informed"): NOTE,
}


def flag_class(flag: Flag) -> FlagClass:
    """`flag`'s class, from its action. An Override the gate marks unfixable blocks, as G2, G3 and
    G5 decide their own verdicts: G5's no_mode suggests a pick for the user, but its model is
    rejected."""
    match flag.action.kind:
        case "reject":
            return "blocking"
        case "override":
            return "fixable" if flag.fixable else "blocking"
        case "keep":
            return "note"
        case "exclude_traces" | "exclude_record":
            return "automatic"
