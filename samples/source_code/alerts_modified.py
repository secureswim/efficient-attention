"""Alert severity and display rules."""


class Alert:
    def severity(self, error_count):
        """Convert a recent error count to a severity label."""
        if error_count >= 10:
            return "CRITICAL"
        return "NORMAL"

    def display_title(self, service_name):
        """Build the title shown in a notification."""
        return f"Incident notice: {service_name}"
