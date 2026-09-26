import pytest
from forge_bench_slot1 import main

# Reference implementation for converting integer to English words (0-9999999)
def number_to_words(n):
    ones = [
        'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
        'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen']
    tens = [
        '', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']
    def word_upto_999(num):
        if num < 20:
            return ones[num]
        elif num < 100:
            q, r = divmod(num, 10)
            return tens[q] if r == 0 else f"{tens[q]} {ones[r]}"
        else:
            q, r = divmod(num, 100)
            if r == 0:
                return f"{ones[q]} hundred"
            return f"{ones[q]} hundred {word_upto_999(r)}"
    if n < 1000:
        return word_upto_999(n)
    elif n < 1_000_000:
        q, r = divmod(n, 1000)
        head = f"{word_upto_999(q)} thousand"
        if r == 0:
            return head
        return f"{head} {word_upto_999(r)}"
    else:
        q, r = divmod(n, 1_000_000)
        head = f"{word_upto_999(q)} million"
        if r == 0:
            return head
        # r < 1_000_000
        if r < 1000:
            return f"{head} {word_upto_999(r)}"
        t, u = divmod(r, 1000)
        if u == 0:
            return f"{head} {word_upto_999(t)} thousand"
        return f"{head} {word_upto_999(t)} thousand {word_upto_999(u)}"

@pytest.mark.parametrize("num_str,num_value", [
    ("0", 0),
    ("8", 8),
    ("14", 14),
    ("20", 20),
    ("50", 50),
    ("23", 23),
    ("101", 101),
    ("999", 999),
    ("1001", 1001),
    ("123456", 123456),
    ("9999999", 9999999),
    ("7000000", 7000000),
    ("1000000", 1000000),
])
def test_valid_english_outputs(capsys, num_str, num_value):
    expected = number_to_words(num_value) + '\n'
    exitcode = main([num_str])
    out = capsys.readouterr()
    assert exitcode == 0
    assert out.out == expected
    assert out.out.count('\n') == 1
    assert out.err == ''

@pytest.mark.parametrize("arg", [
    "",           # empty
    "-1",         # negative
    "abc",        # not numeric
    "012",        # leading zero
    "0000",       # multiple zeros
    "1.0",        # float
    " 5",         # leading space
    "5 ",         # trailing space
    "10000000",   # just over max
    "99999999",   # way over max
    "1e2",        # exponent
    "0x10",       # hex notation
    "+1",         # plus sign not accepted
    "01",         # leading zero
])
def test_invalid_inputs_report_error(capsys, arg):
    exitcode = main([arg])
    out = capsys.readouterr()
    assert exitcode == 1
    assert out.out == 'error\n'
    assert out.out.count('\n') == 1
    assert out.err == ''

@pytest.mark.parametrize("argv,reason", [
    ([], "no arguments"),
    (["123", "456"], "too many arguments"),
    ([""], "empty argument"),
])
def test_argument_count_contract(capsys, argv, reason):
    exitcode = main(argv)
    out = capsys.readouterr()
    assert exitcode == 1
    assert out.out == 'error\n'
    assert out.out.count('\n') == 1
    assert out.err == ''

@pytest.mark.parametrize("zero_arg,should_succeed", [
    ("0", True),
    ("00", False),
    ("000", False),
])
def test_zero_and_leading_zeros(capsys, zero_arg, should_succeed):
    exitcode = main([zero_arg])
    out = capsys.readouterr()
    if should_succeed:
        assert exitcode == 0
        assert out.out == 'zero\n'
    else:
        assert exitcode == 1
        assert out.out == 'error\n'
    assert out.out.count('\n') == 1
    assert out.err == ''
