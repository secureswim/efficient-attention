"""Export file names and row formatting."""

DEFAULT_ENCODING = "utf-8"


def export_extension(file_type):
    """Choose the file extension for an export."""
    if file_type == "table":
        return "csv"
    return "json"


def row_caption(row_number):
    """Create a human-readable caption for a row."""
    return f"Record {row_number}"
