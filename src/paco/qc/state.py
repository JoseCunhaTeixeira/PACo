"""The run's state (S3 of PACo's agent guidelines): rebuilt from its QC log and its files, one
way, whatever was saved before."""

from pathlib import Path

from sigpipe.masw.runs import RunError, RunManifest
from sigpipe.masw.runs.writing import write_atomic

from paco.qc.config import CONFIG_FILE, QCConfig
from paco.qc.models import Budgets
from paco.qc.report import REPORT_FILE, QCReport, build_report


def rebuild_state(run_folder: Path) -> QCReport:
    """The run's state (S3), rebuilt from its QC log and its files, whatever was saved before:
    each unit's attempts, verdicts and flags, whose each window's work is (a person's in PAC,
    verified by them), the retries spent. The run's ID and window count from its run.json, the
    budgets from its qc_config.json; a run without them, from its saved report."""
    manifest_path, config_path = run_folder / "run.json", run_folder / CONFIG_FILE
    saved = run_folder / REPORT_FILE
    if manifest_path.exists():
        manifest = RunManifest.model_validate_json(manifest_path.read_text())
        run_id, n_xmids = manifest.run_id, len(manifest.windows)
    elif saved.exists():
        before = QCReport.model_validate_json(saved.read_text())
        run_id, n_xmids = before.run_id, before.n_xmids
    else:
        raise RunError(f"{run_folder} holds no run: no run.json, no QC report.")
    if config_path.exists():
        budgets = QCConfig.model_validate_json(config_path.read_text()).budgets
    elif saved.exists():
        budgets = QCReport.model_validate_json(saved.read_text()).budgets
    else:
        budgets = Budgets()
    return build_report(run_id, run_folder, budgets, n_xmids)


def read_report(run_folder: Path) -> QCReport:
    """The run's report: its state rebuilt (`rebuild_state`), never read from a file that a
    change in PAC (a curve picked, a window inverted by hand) may have left behind; the file,
    a snapshot of it for PAC and people, written again when it differs."""
    report = rebuild_state(run_folder)
    path = run_folder / REPORT_FILE
    text = report.model_dump_json(indent=2)
    if not path.exists() or path.read_text() != text:
        write_atomic(path, text)
    return report
