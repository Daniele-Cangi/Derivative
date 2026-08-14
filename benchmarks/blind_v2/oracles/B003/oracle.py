from datetime import datetime, timezone

import pytest

from library.core import parse_rfc3339


def test_rfc3339_offsets_are_normalized_to_utc():
    assert parse_rfc3339("1996-12-19T16:39:57-08:00") == datetime(
        1996,
        12,
        20,
        0,
        39,
        57,
        tzinfo=timezone.utc,
    )
    parsed = parse_rfc3339("1985-04-12T23:20:50.520Z")
    assert parsed == datetime(1985, 4, 12, 23, 20, 50, 520000, tzinfo=timezone.utc)
    assert parsed.tzinfo is timezone.utc


@pytest.mark.parametrize(
    "value",
    ["2026-08-14T12:00:00", "2026-08-14T24:00:00Z", "not-a-timestamp"],
)
def test_invalid_profile_values_are_rejected(value):
    with pytest.raises(ValueError):
        parse_rfc3339(value)
