"""Account labels and notices."""

from datetime import date


class Account:
    category = "member"

    def tier(self, credits):
        """Return the account tier."""
        if credits >= 100:
            return "GOLD"
        return "BASIC"

    def renewal_note(self, year):
        """Return a short renewal notice."""
        return f"Your account renews in February {year}."
