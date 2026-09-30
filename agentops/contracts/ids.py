import uuid
from datetime import UTC, datetime


def new_id(prefix: str) -> str:
    """Short readable id, e.g. ev_3fa91c2b. Enough entropy for one run."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def utc_now() -> datetime:
    """Always timezone-aware UTC. Naive datetimes cause silent bugs."""
    return datetime.now(UTC)
