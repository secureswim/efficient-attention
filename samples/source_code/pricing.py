"""Simple checkout rules."""

TAX_REGION = "local"


def shipping_label(subtotal):
    """Name the shipping tier from the subtotal."""
    if subtotal >= 50:
        return "FREE"
    return "STANDARD"


def receipt_footer(store_name):
    """Build the text printed below a receipt."""
    return f"Thank you for shopping at {store_name}."
