"""
Shared time range utilities for API endpoints.

Provides a consistent interface for computing date ranges from a `days_back`
parameter, ensuring every endpoint that accepts a time selector produces
identical start/end boundaries.
"""

from datetime import datetime, timedelta
from typing import Tuple


def get_time_range(days_back: int) -> Tuple[datetime, datetime]:
    """
    Compute a (start_date, end_date) tuple from a days_back integer.

    Args:
        days_back: Number of days to look back from now.

    Returns:
        Tuple of (start_date, end_date) where end_date is UTC now.
    """
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days_back)
    return start_date, end_date
