"""Background jobs: long work (an inversion) runs in this process while the tools keep answering.

The agent gets a job ID at once, and follows the job through its record on disk.
"""

from .manager import JobManager

__all__ = ["JobManager"]
