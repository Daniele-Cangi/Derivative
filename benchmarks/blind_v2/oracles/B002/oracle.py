import pytest

from library.core import resolve_pointer


def test_rfc6901_pointer_evaluation():
    document = {
        "": "empty-key",
        "a/b": 1,
        "m~n": 2,
        "items": ["zero", {"enabled": True}],
    }
    assert resolve_pointer(document, "") is document
    assert resolve_pointer(document, "/") == "empty-key"
    assert resolve_pointer(document, "/a~1b") == 1
    assert resolve_pointer(document, "/m~0n") == 2
    assert resolve_pointer(document, "/items/1/enabled") is True


@pytest.mark.parametrize("pointer", ["items/0", "/a~2b", "/items/01", "/items/9", "/missing"])
def test_invalid_or_unresolvable_pointer_is_rejected(pointer):
    with pytest.raises(ValueError):
        resolve_pointer({"items": [1, 2]}, pointer)
