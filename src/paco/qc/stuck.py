"""A request that cannot go on without the user: the agent asks them, with the options the
message gives."""

from sigpipe.masw.runs import RunError


class Stuck(RunError):
    """The request cannot go on: the user chooses what to try, among the options the message
    says (a window length that fits, a new run, stopping)."""
