import io
import os
import sys
import tempfile
import pytest
from reverse_chunks import main

def reference_reverse_chunks(text, chunk_size):
    # Build reference output: decode to Unicode, split into fixed-size chunks, reverse, join
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return ''.join(chunks[::-1])

@pytest.fixture
def sample_unicode_file():
    # Contains ASCII and multi-byte Unicode codepoints
    decoded = 'abc\u03a9\u03b4\u03b5\u03b69'  # a b c Ω δ ε η 9
    tf = tempfile.NamedTemporaryFile(delete=False, mode='wb')
    tf.write(decoded.encode('utf-8'))
    tf.close()
    try:
        yield tf.name, decoded
    finally:
        os.unlink(tf.name)

@pytest.fixture
def empty_file():
    tf = tempfile.NamedTemporaryFile(delete=False, mode='wb')
    tf.close()
    try:
        yield tf.name
    finally:
        os.unlink(tf.name)

@pytest.fixture
def invalid_utf8_file():
    tf = tempfile.NamedTemporaryFile(delete=False, mode='wb')
    tf.write(b'abc\xffdef')
    tf.close()
    try:
        yield tf.name
    finally:
        os.unlink(tf.name)

# Helper for capturing output
class CLIRedirect:
    def __init__(self):
        self._stdout_orig = sys.stdout
        self._stderr_orig = sys.stderr
        self.bytesio = io.BytesIO()
        self.strio = io.StringIO()
        self.wrapper = io.TextIOWrapper(self.bytesio, encoding='utf-8')

    def __enter__(self):
        sys.stdout = self.wrapper
        sys.stderr = self.strio
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.flush()
        sys.stdout = self._stdout_orig
        sys.stderr = self._stderr_orig

    @property
    def stdout(self):
        self.wrapper.flush()
        return self.bytesio.getvalue()
    @property
    def stderr(self):
        return self.strio.getvalue()

# TEST 1: Direct public target call with normal input, reversed chunks
# Args: [filename, chunk_size]
def test_chunk_reverse_regular(sample_unicode_file):
    filename, decoded = sample_unicode_file
    chunk_size = 3
    args = [filename, str(chunk_size)]
    expected_out = reference_reverse_chunks(decoded, chunk_size).encode('utf-8')
    with CLIRedirect() as redir:
        code = main(args)
    assert code == 0
    assert redir.stdout == expected_out
    assert redir.stderr == ''

def test_invalid_chunk_nonint(sample_unicode_file):
    filename, _ = sample_unicode_file
    args = [filename, 'notanint']
    with CLIRedirect() as redir:
        code = main(args)
    assert code == 1
    assert redir.stdout == b''
    assert redir.stderr == 'error: invalid input'

def test_empty_file_no_output(empty_file):
    args = [empty_file, '10']
    with CLIRedirect() as redir:
        code = main(args)
    assert code == 0
    assert redir.stdout == b''
    assert redir.stderr == ''

def test_chunk_size_too_large(sample_unicode_file):
    filename, decoded = sample_unicode_file
    chunk_size = len(decoded) + 7
    args = [filename, str(chunk_size)]
    with CLIRedirect() as redir:
        code = main(args)
    assert code == 0
    assert redir.stdout == b''
    assert redir.stderr == ''

def test_zero_chunk(sample_unicode_file):
    filename, _ = sample_unicode_file
    args = [filename, '0']
    with CLIRedirect() as redir:
        code = main(args)
    assert code == 1
    assert redir.stdout == b''
    assert redir.stderr == 'error: invalid input'

def test_negative_chunk_size(sample_unicode_file):
    filename, _ = sample_unicode_file
    args = [filename, '-2']
    with CLIRedirect() as redir:
        code = main(args)
    assert code == 1
    assert redir.stdout == b''
    assert redir.stderr == 'error: invalid input'

def test_too_few_args(sample_unicode_file):
    filename, _ = sample_unicode_file
    args = [filename]
    with CLIRedirect() as redir:
        code = main(args)
    assert code == 1
    assert redir.stdout == b''
    assert redir.stderr == 'error: invalid input'

def test_too_many_args(sample_unicode_file):
    filename, _ = sample_unicode_file
    args = [filename, '2', 'extra']
    with CLIRedirect() as redir:
        code = main(args)
    assert code == 1
    assert redir.stdout == b''
    assert redir.stderr == 'error: invalid input'

def test_invalid_utf8_file(invalid_utf8_file):
    args = [invalid_utf8_file, '2']
    with CLIRedirect() as redir:
        code = main(args)
    assert code == 1
    assert redir.stdout == b''
    assert redir.stderr == 'error: invalid input'

def test_multibyte_unicode():
    sample = '\U0001F34E\U0001F34A\U0001F34Bxyz'  # 3 emoji + x y z
    tf = tempfile.NamedTemporaryFile(delete=False, mode='wb')
    tf.write(sample.encode('utf-8'))
    tf.close()
    try:
        args = [tf.name, '2']
        expected = reference_reverse_chunks(sample, 2).encode('utf-8')
        with CLIRedirect() as redir:
            code = main(args)
        assert code == 0
        assert redir.stdout == expected
        assert redir.stderr == ''
    finally:
        os.unlink(tf.name)

def test_leading_zeros_chunk_size(sample_unicode_file):
    filename, decoded = sample_unicode_file
    args = [filename, '0003']
    expected = reference_reverse_chunks(decoded, 3).encode('utf-8')
    with CLIRedirect() as redir:
        code = main(args)
    assert code == 0
    assert redir.stdout == expected
    assert redir.stderr == ''

def test_nonexistent_file():
    name = 'f__notexists_%d' % os.getpid()
    args = [name, '2']
    with CLIRedirect() as redir:
        code = main(args)
    assert code == 1
    assert redir.stdout == b''
    assert redir.stderr == 'error: invalid input'
