"""Inventory labels."""


def stock_status(quantity):
    """Classify the number of units on hand."""
    if quantity == 0:
        return "OUT"
    if quantity < 5:
        return "LOW"
    return "READY"


def bin_label(aisle, shelf):
    """Format a physical storage location."""
    return f"Aisle {aisle}, shelf {shelf}"
