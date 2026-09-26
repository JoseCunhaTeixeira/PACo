"""The short description of a run that the agent reads. Runs themselves are sigpipe's
(sigpipe.masw.runs): processed, found and read there."""

from pydantic import BaseModel, ConfigDict
from sigpipe.masw.runs import RunManifest

# Recorded in a run's manifest with sigpipe's version.
PACKAGES = ("paco",)


class RunSummary(BaseModel):
    """Short description of a run for the agent."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    profile: str
    preset: str
    path: str  # run folder, relative to the output directory
    n_windows: int
    # Windows with a dispersion image: not judged yet (a model reads "n_succeeded" as good
    # windows).
    n_processed: int
    n_failed: int
    n_skipped: int  # positions without any valid shot (active profiles only)
    duration_s: float
    errors: tuple[str, ...]  # a few distinct failures: "xmid 12.50: <type>: <message>"


_MAX_ERRORS = 3
_MAX_ERROR_LENGTH = 200


def summarize_run(manifest: RunManifest) -> RunSummary:
    failed = [window for window in manifest.windows if window.status == "failed"]

    # Distinct messages only: one cause failing every window is one error, not a hundred.
    errors: list[str] = []
    seen: set[str] = set()
    for window in failed:
        if window.error is None or window.error in seen:
            continue
        seen.add(window.error)
        errors.append(f"xmid {window.xmid:.2f}: {window.error[:_MAX_ERROR_LENGTH]}")
        if len(errors) == _MAX_ERRORS:
            break

    return RunSummary(
        run_id=manifest.run_id,
        profile=manifest.profile.name,
        preset=manifest.preset.mode,
        path=f"{manifest.profile.name}/{manifest.run_id}",
        n_windows=len(manifest.windows),
        n_processed=len(manifest.windows) - len(failed),
        n_failed=len(failed),
        n_skipped=manifest.n_positions - len(manifest.windows),
        duration_s=round((manifest.finished_at - manifest.started_at).total_seconds(), 1),
        errors=tuple(errors),
    )
