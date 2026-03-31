"""
Business rule constants for the order processing system.
"""

# Discount rate tiers (applied to gross order total)
DISCOUNT_RATES = {
    "standard": 0.05,   # 5% for standard members
    "premium":  0.10,   # 10% for premium members
    "vip":      0.15,   # 15% for VIP members
}

# Minimum order value (dollars) required to qualify for a discount
MINIMUM_ORDER_FOR_DISCOUNT: float = 10.00

# Currency precision (number of decimal places)
CURRENCY_DECIMALS: int = 2

# Maximum number of line items allowed per order
MAX_LINE_ITEMS: int = 50
