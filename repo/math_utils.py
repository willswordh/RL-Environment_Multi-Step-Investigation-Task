"""
Shared math utility functions used across the order processing system.
"""


def round_currency(value: float, decimals: int = 2) -> float:
    """Round a monetary value to the given number of decimal places.

    Used throughout the system whenever a dollar amount needs to be
    rounded to cents.

    Args:
        value:    The floating-point amount to round.
        decimals: Number of decimal places (default 2 for cents).

    Returns:
        The rounded value as a float.
    """
    return round(value, decimals)


def percentage_of(value: float, percent: float) -> float:
    """Return `percent`% of `value`.

    Args:
        value:   The base amount.
        percent: The percentage to apply (e.g. 10 for 10%).

    Returns:
        The computed percentage as a float.
    """
    return value * (percent / 100.0)


def clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp `value` to the inclusive range [minimum, maximum]."""
    return max(minimum, min(maximum, value))
