"""The code the autonomous agent's results depend on: sigpipe's commit, as the installer recorded
it (uv.lock pins it), and PACo's quality control, hashed, now that the agent is part of PACo."""

import hashlib
import json
from importlib.metadata import distribution
from pathlib import Path

from paco import qc


def commit(name: str) -> str | None:
    """The git commit distribution `name` was installed from, as the installer recorded it (PEP
    610's direct_url.json); None when it came from elsewhere, such as an index or a folder."""
    text = distribution(name).read_text("direct_url.json")
    if text is None:
        return None
    found = json.loads(text).get("vcs_info", {}).get("commit_id")
    return found if isinstance(found, str) else None


def paco_code() -> str:
    """PACo's quality control (paco.qc, which G1 and the other gates run), its sources hashed: a
    change of them makes other keys."""
    root = Path(qc.__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
