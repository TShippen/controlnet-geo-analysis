"""How a result says that the evidence did not support a derived value.

A derived value is either reported with the support behind it or withheld
with the reason. It is never reported with a caveat attached.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Withheld:
    """A derived value that the evidence did not support.

    Attributes:
        reason: Why the value was not derived, phrased for the reader of the
            result with no leading capital and no closing period.
    """

    reason: str
