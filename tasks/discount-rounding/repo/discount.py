"""
Discount calculation logic for the order processing system.
"""
from repo.config import DISCOUNT_RATES, MINIMUM_ORDER_FOR_DISCOUNT
from repo import math_utils


def get_discount_rate(membership_tier: str) -> float:
    """Look up the discount rate for a given membership tier.

    Args:
        membership_tier: One of "standard", "premium", or "vip".

    Returns:
        The fractional discount rate (e.g. 0.10 for 10%).

    Raises:
        ValueError: If the membership tier is not recognised.
    """
    tier = membership_tier.lower()
    if tier not in DISCOUNT_RATES:
        raise ValueError(
            f"Unknown membership tier '{membership_tier}'. "
            f"Valid tiers: {list(DISCOUNT_RATES.keys())}"
        )
    return DISCOUNT_RATES[tier]


def apply_discount(gross_total: float, membership_tier: str) -> float:
    """Apply a membership discount to a gross order total.

    If the gross total is below MINIMUM_ORDER_FOR_DISCOUNT, no
    discount is applied and the original total is returned unchanged.

    The discount rate is looked up via get_discount_rate(), the
    discounted amount is computed, and the result is rounded to the
    nearest cent using math_utils.round_currency().

    Args:
        gross_total:      Pre-discount order total in dollars.
        membership_tier:  Customer membership tier.

    Returns:
        The post-discount total, rounded to two decimal places.
    """
    if gross_total < MINIMUM_ORDER_FOR_DISCOUNT:
        return math_utils.round_currency(gross_total)

    rate = get_discount_rate(membership_tier)
    discounted = gross_total * (1.0 - rate)
    return math_utils.round_currency(discounted)


def calculate_savings(gross_total: float, membership_tier: str) -> float:
    """Return the dollar amount saved by the membership discount.

    Args:
        gross_total:     Pre-discount order total in dollars.
        membership_tier: Customer membership tier.

    Returns:
        Savings in dollars, rounded to two decimal places.
    """
    discounted = apply_discount(gross_total, membership_tier)
    savings = gross_total - discounted
    return math_utils.round_currency(savings)
