import pytest
import datetime
from itertools import islice
from your_module import your_function  # Replace with the actual module and function names

def test_simple_contiguous_ranges():
    dates = (datetime.date(2023, 1, d) for d in [1, 2, 3, 5, 6, 10])
    result = list(your_function(dates))
    expected = [
        (datetime.date(2023, 1, 1), datetime.date(2023, 1, 3)),
        (datetime.date(2023, 1, 5), datetime.date(2023, 1, 6)),
        (datetime.date(2023, 1, 10), datetime.date(2023, 1, 10))
    ]
    assert result == expected

def test_empty_generator():
    dates = iter(())
    result = list(your_function(dates))
    assert result == []

def test_single_date():
    dates = iter([datetime.date(2023, 2, 15)])
    result = list(your_function(dates))
    assert result == [(datetime.date(2023, 2, 15), datetime.date(2023, 2, 15))]

def test_non_date_input_raises():
    dates = iter([datetime.date(2023, 3, 1), "not a date"])
    with pytest.raises(ValueError):
        list(your_function(dates))

def infinite_date_generator(start_date):
    current = start_date
    while True:
        yield current
        current += datetime.timedelta(days=1)

def test_infinite_generator_first_n_ranges():
    # Should not consume memory infinitely and produce expected first n ranges.
    gen = infinite_date_generator(datetime.date(2023, 1, 1))
    # Modify generator to skip some dates to create distinct ranges
    def modified_gen():
        d = datetime.date(2023, 1, 1)
        while True:
            yield d
            # Skip every 10th day to break continuity every 10 days
            if (d - datetime.date(2023, 1, 1)).days % 10 == 8:
                d += datetime.timedelta(days=2)
            else:
                d += datetime.timedelta(days=1)

    ranges_iter = your_function(modified_gen())
    first_ranges = list(islice(ranges_iter, 3))

    assert all(
        isinstance(start, datetime.date) and isinstance(end, datetime.date) and start <= end
        for start, end in first_ranges
    )

    # Check the first range is 9 days long (days 1-9 inclusive)
    assert (first_ranges[0][1] - first_ranges[0][0]).days == 8

    # Each range should not overlap
    for i in range(len(first_ranges) - 1):
        assert first_ranges[i][1] < first_ranges[i+1][0] - datetime.timedelta(days=1)
