"""
Data models for the order processing system.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class LineItem:
    """A single line item in a customer order."""
    name: str
    unit_price: float   # price per unit in dollars
    quantity: int

    @property
    def subtotal(self) -> float:
        return self.unit_price * self.quantity


@dataclass
class Order:
    """A customer order containing one or more line items."""
    order_id: str
    customer: str
    items: List[LineItem] = field(default_factory=list)

    @property
    def gross_total(self) -> float:
        """Sum of all line item subtotals before any discount."""
        return sum(item.subtotal for item in self.items)
