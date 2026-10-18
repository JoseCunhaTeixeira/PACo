"""The lines the A/B evaluation plays on (docs/autonomous_plan.md §12): active_test_profile, as the input folder
holds it, and the defect copies PACo's suite makes of it (paco.evaluation.defects): active_dead,
its first record's trace 40 zeroed, and active_refracted, a head wave nearly as strong as the
surface waves (0.8 of each trace's peak). Written once per evaluation, in its folder."""

from pathlib import Path

from paco.autonomous.upstream import defects

# The real lines both agents play, as the input folder holds them.
LINES = ("active_test_profile",)
# The defect copies: the copy, its line, and the trace zeroed or the head wave's velocity (m/s)
# and its amplitude over each trace's peak (PACo's).
DEAD_TRACE = ("active_dead", "active_test_profile", 40)
REFRACTED = ("active_refracted", "active_test_profile", 1200.0, 0.8)


def build_inputs(source: Path, target: Path) -> Path:
    """`target`, holding the lines of `source` (linked, not copied: active_test_profile's records weigh some
    220 MB) and the defect copies of active_test_profile; built once, kept when it exists."""
    for name in LINES:
        if (source / name).is_dir() and not (target / name).exists():
            (target / name).mkdir(parents=True)
            for path in (source / name).iterdir():
                (target / name / path.name).symlink_to(path.resolve())
    variant, origin, trace = DEAD_TRACE
    if (source / origin).is_dir() and not (target / variant).exists():
        defects.dead_trace(source / origin, target / variant, trace)
    variant, origin, velocity, ratio = REFRACTED
    if (source / origin).is_dir() and not (target / variant).exists():
        defects.refracted(source / origin, target / variant, velocity, ratio)
    return target


def offered(source: Path) -> frozenset[str]:
    """The lines build_inputs makes of `source`: its real ones, and the defect copies of
    active_test_profile when it holds it."""
    lines = {name for name in LINES if (source / name).is_dir()}
    lines |= {variant for variant, origin, *_ in (DEAD_TRACE, REFRACTED) if origin in lines}
    return frozenset(lines)
