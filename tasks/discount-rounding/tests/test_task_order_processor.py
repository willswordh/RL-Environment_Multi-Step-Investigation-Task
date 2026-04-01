"""
Tests for the order processing pipeline.
"""
import sys
import os
import pytest

# Allow imports from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from repo.models import Order, LineItem
from repo.order_processor import process_order


class TestDiscountRounding:
    """Tests that verify correct rounding of discounted order totals."""

    def test_premium_discount_rounding(self):
        """Premium discount case that validates rounded cents.

        Order:  1 x $24.75  →  gross = $24.75
        10% discount:  $24.75 * 0.90 = $22.275
        Expected final total: $22.28
        """
        order = Order(
            order_id="ORD-001",
            customer="Alice",
            items=[LineItem(name="Widget", unit_price=24.75, quantity=1)],
        )
        result = process_order(order, membership_tier="premium")
        assert result["final_total"] == 22.28

    def test_premium_discount_rounding_half_up(self):
        """Premium discount case that should preserve an exact-cent result.

        Order:  3 x $7.50  →  gross = $22.50
        10% discount:  $22.50 * 0.90 = $20.25
        Expected final total: $20.25
        """
        order = Order(
            order_id="ORD-002",
            customer="Bob",
            items=[LineItem(name="Gadget", unit_price=7.50, quantity=3)],
        )
        result = process_order(order, membership_tier="premium")
        assert result["final_total"] == 20.25

    def test_vip_discount_rounding(self):
        """VIP discount case with a cent-level boundary condition."""
        order = Order(
            order_id="ORD-003",
            customer="Carol",
            items=[LineItem(name="Doohickey", unit_price=24.50, quantity=1)],
        )
        result = process_order(order, membership_tier="vip")
        assert result["final_total"] == 20.83, (
            f"Expected final_total=20.83 but got {result['final_total']}."
        )

    def test_standard_discount_no_rounding_issue(self):
        """A simple case where rounding mode does not matter (exact cents).

        Order:  2 x $10.00  →  gross = $20.00
        5% discount:  $20.00 * 0.95 = $19.00  (exact)
        Expected: $19.00
        """
        order = Order(
            order_id="ORD-004",
            customer="Dave",
            items=[LineItem(name="Thingamajig", unit_price=10.00, quantity=2)],
        )
        result = process_order(order, membership_tier="standard")
        assert result["final_total"] == 19.00

    def test_below_minimum_no_discount_applied(self):
        """Orders below the minimum threshold should not receive a discount."""
        order = Order(
            order_id="ORD-005",
            customer="Eve",
            items=[LineItem(name="Cheap Item", unit_price=5.00, quantity=1)],
        )
        result = process_order(order, membership_tier="premium")
        assert result["final_total"] == 5.00

    def test_savings_calculation(self):
        """Savings should equal gross minus discounted total."""
        order = Order(
            order_id="ORD-006",
            customer="Frank",
            items=[LineItem(name="Big Widget", unit_price=100.00, quantity=1)],
        )
        result = process_order(order, membership_tier="vip")
        assert result["savings"] == pytest.approx(15.00, abs=0.01)
