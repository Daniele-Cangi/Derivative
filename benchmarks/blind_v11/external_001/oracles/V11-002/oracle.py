import pytest
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
from palindrome_tagger_cli import main

# Test 1: Successful processing with palindromes and non-palindromes
# Input: 'level,river,radar,deed,cat'
# Output: 'LEVEL,river,RADAR,DEED,cat'
def test_palindrome_and_non_palindrome_stdout_and_exit():
    argv = ['progname', 'level,river,radar,deed,cat']
    out = StringIO()
    with redirect_stdout(out):
        exit_code = main(argv)
    assert out.getvalue() == 'LEVEL,river,RADAR,DEED,cat\n'
    assert exit_code == 0

# Test 2: Error when there is any empty word in the comma-separated string
@pytest.mark.parametrize("input_arg", [
    'cat,,mat',   # middle empty
    ',bat,cat',   # leading empty
    'bat,cat,',   # trailing empty
    ',',          # two empties
    '',           # single empty word
    'a,,b',       # middle empty, non-palindrome
])
def test_error_on_empty_word(input_arg):
    argv = ['progname', input_arg]
    out = StringIO()
    with redirect_stdout(out):
        exit_code = main(argv)
    assert out.getvalue() == 'ERROR\n'
    assert exit_code == 2

# Test 3: Error when no positional argument is given
def test_error_on_missing_argument():
    argv = ['progname']  # argv[0] only
    out = StringIO()
    with redirect_stdout(out):
        exit_code = main(argv)
    assert out.getvalue() == 'ERROR\n'
    assert exit_code == 2

# Test 4: All single-character words are palindromes, each must be uppercased
# Input: 'a,b,c'
# Output: 'A,B,C'
def test_single_char_words_all_palindromes():
    argv = ['progname', 'a,b,c']
    out = StringIO()
    with redirect_stdout(out):
        exit_code = main(argv)
    assert out.getvalue() == 'A,B,C\n'
    assert exit_code == 0

# Test 5: All words are not palindromes, no change in casing
# Input: 'dog,cat,bat'
# Output: 'dog,cat,bat'
def test_no_palindromes_all_unchanged():
    argv = ['progname', 'dog,cat,bat']
    out = StringIO()
    with redirect_stdout(out):
        exit_code = main(argv)
    assert out.getvalue() == 'dog,cat,bat\n'
    assert exit_code == 0

# Test 6: Mix of palindromes and non-palindromes at start/end/middle
# Input: 'eye,madam,kayak,hello'
# Output: 'EYE,MADAM,KAYAK,hello'
def test_palindromes_and_non_palindromes_mixed():
    argv = ['progname', 'eye,madam,kayak,hello']
    out = StringIO()
    with redirect_stdout(out):
        exit_code = main(argv)
    assert out.getvalue() == 'EYE,MADAM,KAYAK,hello\n'
    assert exit_code == 0

# Test 7: Output must contain only ASCII characters and no spaces, with palindromes at ends
# Input: 'stats,tenet,apple'
# Output: 'STATS,TENET,apple'
def test_ascii_and_format_no_spaces():
    argv = ['progname', 'stats,tenet,apple']
    out = StringIO()
    with redirect_stdout(out):
        exit_code = main(argv)
    output = out.getvalue()
    assert output == 'STATS,TENET,apple\n'
    assert all(ord(c) < 128 for c in output)
    assert ' ' not in output
    assert exit_code == 0

# Test 8: No output must ever be written to stderr on any code path
@pytest.mark.parametrize(
    "argv, expected_out, expected_status",
    [
        (['progname'], 'ERROR\n', 2),
        (['progname', ''], 'ERROR\n', 2),
        (['progname', 'madam,otto,repaper'], 'MADAM,OTTO,REPAPER\n', 0),
        (['progname', 'apple'], 'apple\n', 0),
    ]
)
def test_no_stderr_any_path(argv, expected_out, expected_status):
    out = StringIO()
    err = StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        exit_code = main(argv)
    assert out.getvalue() == expected_out
    assert err.getvalue() == ''
    assert exit_code == expected_status
