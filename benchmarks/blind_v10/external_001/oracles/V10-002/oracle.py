import pytest
from forge_bench_slot2 import main
import sys

# Helper function to compute ISO weekday from YYYYMMDD string for fixture creation
import datetime

def iso_weekday_name(yyyymmdd):
    year = int(yyyymmdd[:4])
    month = int(yyyymmdd[4:6])
    day = int(yyyymmdd[6:8])
    dt = datetime.date(year, month, day)
    return dt.strftime('%A')

@pytest.mark.parametrize(
    "input_date",
    [
        "20221005",   # Wednesday
        "19000101",   # Monday (min bound)
        "20991231",   # Thursday (max bound)
        "20000229",   # Leap day, valid (Tuesday)
        "19991231",   # Friday
    ],
)
def test_valid_dates_stdout_and_exit(monkeypatch, capsys, input_date):
    # All of these are valid dates within range
    expected_weekday = iso_weekday_name(input_date)
    # Call the CLI entrypoint as required, passing only the user argument
    res = main([input_date])
    assert res == 0
    cap = capsys.readouterr()
    assert cap.out == expected_weekday + '\n'
    assert cap.err == ''

@pytest.mark.parametrize(
    "bad_arg",
    [
        "20211310",   # Month 13
        "19990231",   # 31st Feb
        "20000230",   # 30th Feb in leap year
        "20991301",   # Month out of bounds
        "19000431",   # April 31st
        "20210229",   # 2021 is not a leap year
        "18991231",   # below lower bound
        "21000101",   # above upper bound
        "abcdefgh",   # Non-numeric
        "202112",     # Too short
        "202201011",  # Too long
        "2021-0101",  # Delimiter included
        "",           # Empty
    ],
)
def test_invalid_dates_stderr_and_exit(monkeypatch, capsys, bad_arg):
    res = main([bad_arg])
    assert res == 1
    cap = capsys.readouterr()
    assert cap.out == 'error\n'
    assert cap.err == ''


def test_missing_and_extra_arguments(monkeypatch, capsys):
    # No arguments
    res0 = main([])
    cap0 = capsys.readouterr()
    assert res0 == 1
    assert cap0.out == 'error\n'
    assert cap0.err == ''

    # More than one argument
    res1 = main(["20220101", "19990101"])
    cap1 = capsys.readouterr()
    assert res1 == 1
    assert cap1.out == 'error\n'
    assert cap1.err == ''
