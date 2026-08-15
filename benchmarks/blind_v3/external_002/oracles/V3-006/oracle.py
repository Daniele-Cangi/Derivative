import pytest
from datetime import timedelta
from parse_time_delta import parse_time_delta

@pytest.mark.parametrize("input_str,expected", [
    ("1d2h30m15s", timedelta(days=1, hours=2, minutes=30, seconds=15)),
    ("2h15s1d30m", timedelta(days=1, hours=2, minutes=30, seconds=15)),
    ("0d0h0m0s", timedelta(days=0)),
    ("3d", timedelta(days=3)),
    ("45m", timedelta(minutes=45)),
    ("2h", timedelta(hours=2)),
    ("15s", timedelta(seconds=15)),
    ("1d", timedelta(days=1)),
    ("1d0h0m0s", timedelta(days=1)),
    ("0h0m15s", timedelta(seconds=15)),
])
def test_valid_inputs(input_str, expected):
    assert parse_time_delta(input_str) == expected

@pytest.mark.parametrize("input_str", [
    "1d2d",          # repeated days
    "1h2h",          # repeated hours
    "15s15s",        # repeated seconds
    "1w",            # unknown unit w
    "1d2x",          # unknown unit x
    "d1",            # missing number before d
    "1d2h30",        # missing unit for 30
    "",              # empty string
    "1dd",           # malformed unit
    "--1d",          # invalid numeric format
    "1d-2h",         # invalid negative value
    "1d 2h",         # space not allowed
    "1 d",           # space between number and unit
    "0d",            # days must appear once or not at all; 0d is allowed but test separately
    "1d0d",          # repeated days even with zero
])
def test_invalid_inputs(input_str):
    with pytest.raises(ValueError):
        parse_time_delta(input_str)

@pytest.mark.parametrize("input_str,expected", [
    ("0d", timedelta(days=0)),         # zero days allowed exactly once
    ("0h0m0s", timedelta(0)),          # zero time without days
    ("0d0h0m0s", timedelta(0)),        # zero time with days zero
])
def test_edge_cases_zero_values(input_str, expected):
    assert parse_time_delta(input_str) == expected

@pytest.mark.parametrize("input_str", [
    "1d2h2h",    # repeated hours
    "1d30m15s15s", # repeated seconds
    "1d1d",      # repeated days
    "1d1h2x",    # unknown unit x
    "d",         # missing number
    "5",         # missing unit
])
def test_more_repeated_and_invalid_units(input_str):
    with pytest.raises(ValueError):
        parse_time_delta(input_str)
