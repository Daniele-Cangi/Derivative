import io
import sys
import pytest
from forge_blind_v9_requirements_slot1 import main

# Utility: simulate sys.stdin, sys.stdout for text (with newlines and bytes)
class IoFixture:
    def __init__(self, bytes_input):
        # input as bytes for stdin (simulate file.read())
        self._inb = io.BytesIO(bytes_input)
        self.stdin = io.TextIOWrapper(self._inb, encoding='utf-8', errors='strict', newline='')
        self.outb = io.BytesIO()
        self.stdout = io.TextIOWrapper(self.outb, encoding='utf-8', newline='')
        self.stderr = io.StringIO()
    def bind(self):
        self._stdin_real = sys.stdin
        self._stdout_real = sys.stdout
        self._stderr_real = sys.stderr
        sys.stdin = self.stdin
        sys.stdout = self.stdout
        sys.stderr = self.stderr
    def unbind(self):
        sys.stdin = self._stdin_real
        sys.stdout = self._stdout_real
        sys.stderr = self._stderr_real
    def done(self):
        self.stdout.flush()
        return self.outb.getvalue().decode('utf-8', 'strict'), self.stderr.getvalue()

# Run main() as CLI with given input bytes (returns (code, stdout, stderr))
def run_main_with_bytes(bytes_input):
    f = IoFixture(bytes_input)
    f.bind()
    try:
        exit_code = main()
    finally:
        out, err = f.done()
        f.unbind()
    return exit_code, out, err

def ref_output_lines(in_lines):
    # Implements the requirement's transformation contract:
    # each input line preserved *if* ASCII alnum (incl empty or \n)
    out_lines = []
    for line in in_lines:
        s = line[:-1] if line.endswith('\n') else line
        if (s == '' or all(('0' <= c <= '9') or ('a' <= c <= 'z') or ('A' <= c <= 'Z') for c in s)):
            out_lines.append(line)
        else:
            # output empty string with newline if input has newline, or just empty string otherwise
            if line.endswith('\n'):
                out_lines.append('\n')
            else:
                out_lines.append('')
    return ''.join(out_lines)

@pytest.mark.parametrize('input_lines', [
    ['abc123\n', 'XYZ\n', '7890\n'],
    ['A\n', 'Z0\n', 'qwerty\n'],
    ['MNBVCXZ1234567890qwertyuioplkjhgfdsazxcvbnm\n'],
])
def test_valid_ascii_alnum_lines_preserved(input_lines):
    # All lines ASCII alnum: output must be identical
    input_bytes = ''.join(input_lines).encode('utf-8')
    code, output, err = run_main_with_bytes(input_bytes)
    ref = ref_output_lines(input_lines)
    assert code == 0
    assert output == ref
    assert err == ''
    # Check output lines exactly match input
    assert output == ''.join(input_lines)

@pytest.mark.parametrize('input_lines', [
    ['abc123\n', 'hello world!\n', '123@#\n', 'XYZ\n', '\n'],
    ['spaces \n', 'UPPERlower\n', 'foo-bar\n'],
    ['nope!\n', '[](){}`~\n', '456\n', '    \n'],
    ['\n', 'word\n', ' \,/.\n', 'Test\n'],
    ['1one\n', '2two?\n', '\n', '&&&&\n', 'A0\n'],
])
def test_mixed_and_invalid_lines(input_lines):
    # Includes both valid and invalid lines: check ref transformation
    input_bytes = ''.join(input_lines).encode('utf-8')
    code, output, err = run_main_with_bytes(input_bytes)
    ref = ref_output_lines(input_lines)
    assert code == 0
    assert output == ref
    assert err == ''
    # Check output has same number of lines as input
    assert output.count('\n') == ''.join(input_lines).count('\n')

@pytest.mark.parametrize('n', [1, 3, 8, 0])
def test_empty_and_all_blank_lines(n):
    # n empty lines, should produce n newlines
    input_lines = ['\n'] * n
    input_bytes = ''.join(input_lines).encode('utf-8')
    code, output, err = run_main_with_bytes(input_bytes)
    assert code == 0
    assert output == ''.join(input_lines)
    assert err == ''
    assert output.count('\n') == n

def test_utf8_decode_error():
    # Contains undecodable bytes, should exit 2 and output nothing
    input_bytes = b'abc\n' + b'\xff\xfe' + b'hello\n'
    code, output, err = run_main_with_bytes(input_bytes)
    assert code == 2
    assert output == ''
    assert err == ''

def test_non_ascii_unicode():
    # Contains valid utf-8, but with non-ASCII unicode chars
    # For example: é (u00e9), Ω (u03a9), ü (u00fc)
    bad_line = 'abc\n' + '\u00e9om\n' + 'XYZ\n'
    bytes_in = bad_line.encode('utf-8')
    code, output, err = run_main_with_bytes(bytes_in)
    assert code == 2
    assert output == ''
    assert err == ''

def test_preserve_line_count_and_trailing_newlines():
    # Lines ending in \n, with legit and invalid lines; all trailing newlines preserved
    input_lines = ['abc\n', 'bad!word\n', '123456\n', '@@@\n']
    input_bytes = ''.join(input_lines).encode('utf-8')
    code, output, err = run_main_with_bytes(input_bytes)
    ref = ref_output_lines(input_lines)
    assert code == 0
    assert output == ref
    # Output lines and input lines must be the same count and structure
    assert output.count('\n') == ''.join(input_lines).count('\n')
    assert err == ''

def test_input_without_final_newline():
    # If input doesn't end with \n, output lines still must correspond. Newlines only present if input line ended with one.
    input_lines = ['Abc', 'D3f', 'bad$', 'X7', '']  # last is empty without newline
    # Make bytes, no newlines except between lines
    input_bytes = '\n'.join(input_lines).encode('utf-8')
    code, output, err = run_main_with_bytes(input_bytes)
    ref = ref_output_lines([l+'\n' for l in input_lines[:-1]] + [input_lines[-1]]) if input_lines[-1] != '' else ref_output_lines([l+'\n' for l in input_lines])
    assert code == 0
    # The output should preserve only those newlines present in the input
    assert output == ref
    assert err == ''

def test_input_of_only_non_ascii_chars():
    # Input is valid UTF-8 but with every line containing only unicode chars
    input_lines = ['Ωmega\n', 'école\n', 'über\n']
    input_bytes = ''.join(input_lines).encode('utf-8')
    code, output, err = run_main_with_bytes(input_bytes)
    assert code == 2
    assert output == ''
    assert err == ''

def test_input_with_tabs_spaces_and_valid_lines():
    # Input with tabs, spaces and valid ascii alnum lines
    input_lines = ['abc123\n', '  tabbed\n', 'XYZ\n', 'space bar\n']
    input_bytes = ''.join(input_lines).encode('utf-8')
    code, output, err = run_main_with_bytes(input_bytes)
    ref = ref_output_lines(input_lines)
    assert code == 0
    assert output == ref
    assert err == ''
    # Spaces and tabs not valid: their lines become empty strings (\n). Valid lines preserved.
    for i, l in enumerate(input_lines):
        s = l.rstrip('\n')
        if (s == '' or not s.isalnum() or not all(ord(ch) < 128 for ch in s)):
            assert output.splitlines(keepends=True)[i] == '\n' if l.endswith('\n') else ''
        else:
            assert output.splitlines(keepends=True)[i] == l

def test_replace_partial_unicode():
    # Input where only one line contains unicode: should trigger error and produce nothing
    input_lines = ['good\n', 'ok123\n', 'badÉ\n', 'fine\n']
    input_bytes = ''.join(input_lines).encode('utf-8', errors='strict')
    code, output, err = run_main_with_bytes(input_bytes)
    assert code == 2
    assert output == ''
    assert err == ''
    # Changing any line with unicode triggers the error
    input_lines2 = ['good\n', 'ok123\n', 'BAD\n', 'fine\n']
    input_bytes2 = ''.join(input_lines2).encode('utf-8', errors='strict')
    code2, output2, err2 = run_main_with_bytes(input_bytes2)
    assert code2 == 0
    assert output2 == ''.join(input_lines2)
    assert err2 == ''
    # Explicitly test that lone invalid content triggers correct lines
    alt_bad = ['BADBADBADBADÉ\n']
    input_bytes3 = ''.join(alt_bad).encode('utf-8', errors='strict')
    code3, output3, err3 = run_main_with_bytes(input_bytes3)
    assert code3 == 2
    assert output3 == ''
    assert err3 == ''
    # But with only ASCII, it's ok
    ok = ['BADBADBADBAD\n']
    code4, output4, err4 = run_main_with_bytes(''.join(ok).encode('utf-8'))
    assert code4 == 0
    assert output4 == ''.join(ok)
    assert err4 == ''
    # This demonstrates rejection is triggered by any non-ASCII char on any line.

# End of tests
