import io
import hashlib
import pytest

from service import hash_stream

class DummyStream(io.BufferedReader):
    """A dummy stream that raises on reading to test error handling."""
    def __init__(self, initial_bytes: bytes):
        self._bytes = initial_bytes
        self._pos = 0

    def read(self, size=-1):
        if self._pos >= len(self._bytes):
            return b''
        if size == -1:
            size = len(self._bytes) - self._pos
        chunk = self._bytes[self._pos:self._pos+size]
        self._pos += size
        return chunk


def test_hash_known_content():
    data = b"OpenAI GPT-4 stream hashing test data."
    stream = io.BufferedReader(io.BytesIO(data))
    expected_hash = hashlib.sha256(data).hexdigest()
    result = hash_stream(stream)
    # Should return expected SHA-256 hex digest
    assert result == expected_hash
    # Stream should not be closed
    assert not stream.closed
    # Stream position advanced to end after reading
    assert stream.tell() == len(data)


def test_hash_large_data():
    # Create 10MB of repeating pattern
    chunk = b"abcd1234" * 128  # 1024 bytes
    repeats = 10240  # 1024 * 10240 = 10MB
    data = chunk * repeats
    stream = io.BufferedReader(io.BytesIO(data))
    expected_hash = hashlib.sha256(data).hexdigest()
    result = hash_stream(stream)
    assert result == expected_hash
    assert not stream.closed
    assert stream.tell() == len(data)


def test_hash_empty_stream_raises():
    empty_stream = io.BufferedReader(io.BytesIO(b""))
    with pytest.raises(ValueError):
        hash_stream(empty_stream)
    # Should not close the stream
    assert not empty_stream.closed


def test_stream_not_closed_and_partial_read():
    data = b"Test data for partial read."
    stream = io.BufferedReader(io.BytesIO(data))
    # Read some bytes before calling hash_stream
    first_bytes = stream.read(10)
    expected_hash = hashlib.sha256(data).hexdigest()
    result = hash_stream(stream)
    # Hashing reads the entire stream from current position
    # So hash of data starting at offset 10
    expected_hash_partial = hashlib.sha256(data[10:]).hexdigest()
    assert result == expected_hash_partial
    # Stream position advanced to end
    assert stream.tell() == len(data)
    # Stream still open
    assert not stream.closed


def test_invalid_type_input():
    # Passing a non BufferedReader should raise TypeError when reading
    with pytest.raises(Exception):
        hash_stream(None)

    with pytest.raises(Exception):
        hash_stream("not a stream")
