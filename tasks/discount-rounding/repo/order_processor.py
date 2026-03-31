"""
Top-level order processing pipeline.
"""
from repo.models import Order
from repo import discount
from repo.config import MAX_LINE_ITEMS


class OrderProcessingError(Exception):
    """Raised when an order cannot be processed."""


def validate_order(order: Order) -> None:
    """Validate an order before processing.

    Args:
        order: The Order to validate.

    Raises:
        OrderProcessingError: If the order fails any validation check.
    """
    if not order.items:
        raise OrderProcessingError(
            f"Order {order.order_id} has no line items."
        )
    if len(order.items) > MAX_LINE_ITEMS:
        raise OrderProcessingError(
            f"Order {order.order_id} exceeds the maximum of "
            f"{MAX_LINE_ITEMS} line items."
        )
    for item in order.items:
        if item.unit_price < 0:
            raise OrderProcessingError(
                f"Line item '{item.name}' has a negative price."
            )
        if item.quantity <= 0:
            raise OrderProcessingError(
                f"Line item '{item.name}' has an invalid quantity: "
                f"{item.quantity}."
            )


def process_order(order: Order, membership_tier: str = "standard") -> dict:
    """Process a customer order and return a summary.

    Steps:
      1. Validate the order.
      2. Compute the gross total from line items.
      3. Apply the membership discount.
      4. Return a result dict with totals and savings.

    Args:
        order:           The customer Order to process.
        membership_tier: Customer membership tier (default "standard").

    Returns:
        A dict with keys:
          - order_id       : str
          - customer       : str
          - gross_total    : float
          - discount_rate  : float
          - final_total    : float
          - savings        : float
    """
    validate_order(order)

    gross_total = order.gross_total
    final_total = discount.apply_discount(gross_total, membership_tier)
    savings = discount.calculate_savings(gross_total, membership_tier)
    rate = discount.get_discount_rate(membership_tier)

    return {
        "order_id":     order.order_id,
        "customer":     order.customer,
        "gross_total":  gross_total,
        "discount_rate": rate,
        "final_total":  final_total,
        "savings":      savings,
    }
