import pytest

from service_module import parse_headers

# Valid input fixture with multiple headers and repeated keys in varied cases
@pytest.fixture
def valid_headers():
    return [
        "Content-Type: text/plain",
        "content-length: 1234",
        "X-Custom-Header: Value1",
        "x-custom-header: Value2",
        "Server: MyServer",
        "Set-Cookie: sessionid=abc123",
        "set-cookie: theme=dark"
    ]

# Invalid headers fixtures
@pytest.fixture
def invalid_headers_missing_colon():
    return [
        "Content-Type text/plain",
        "Accept application/json"
    ]

@pytest.fixture
def invalid_headers_empty_name():
    return [
        ": value",
        " : another"
    ]

@pytest.fixture
def invalid_headers_empty_value():
    return [
        "Host:",
        "X-Test:"
    ]

def test_parse_headers_basic(valid_headers):
    result = parse_headers(valid_headers)
    expected_keys = [
        "content-type",
        "content-length",
        "x-custom-header",
        "server",
        "set-cookie"
    ]
    # Check keys
    assert sorted(result.keys()) == sorted(expected_keys)
    # Check all headers have lists as values
    for values in result.values():
        assert isinstance(values, list)
        assert all(isinstance(v, str) for v in values)
    # Check case insensitivity and normalization
    assert result["content-type"] == ["text/plain"]
    assert result["content-length"] == ["1234"]
    # Check repeated headers collect multiple values preserving order
    assert result["x-custom-header"] == ["Value1", "Value2"]
    assert result["set-cookie"] == ["sessionid=abc123", "theme=dark"]
    # Server header single value
    assert result["server"] == ["MyServer"]

def test_parse_headers_invalid_missing_colon(invalid_headers_missing_colon):
    with pytest.raises(ValueError) as excinfo:
        parse_headers(invalid_headers_missing_colon)
    assert "invalid header line" in str(excinfo.value).lower()

def test_parse_headers_invalid_empty_name(invalid_headers_empty_name):
    with pytest.raises(ValueError) as excinfo:
        parse_headers(invalid_headers_empty_name)
    assert "header name" in str(excinfo.value).lower()

def test_parse_headers_invalid_empty_value(invalid_headers_empty_value):
    with pytest.raises(ValueError) as excinfo:
        parse_headers(invalid_headers_empty_value)
    assert "header value" in str(excinfo.value).lower()

# Edge case: Empty input list returns empty dict
def test_parse_headers_empty_list():
    result = parse_headers([])
    assert result == {}

# Edge case: header names with mixed casing and extra spaces
def test_parse_headers_whitespace_and_casing():
    headers = [
        "  Content-Type :  text/html  ",
        "CONTENT-TYPE: application/json",
        "X-Mixed-Case-HeAdEr : value"
    ]
    result = parse_headers(headers)
    assert result["content-type"] == ["text/html", "application/json"]
    assert result["x-mixed-case-header"] == ["value"]
