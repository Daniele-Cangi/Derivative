import pytest

from library.core import compare_versions


def test_semver_precedence_chain():
    versions = [
        "1.0.0-alpha",
        "1.0.0-alpha.1",
        "1.0.0-alpha.beta",
        "1.0.0-beta",
        "1.0.0-beta.2",
        "1.0.0-beta.11",
        "1.0.0-rc.1",
        "1.0.0",
    ]
    assert all(compare_versions(left, right) == -1 for left, right in zip(versions, versions[1:]))
    assert compare_versions("2.1.0", "2.0.9") == 1


def test_build_metadata_does_not_change_precedence():
    assert compare_versions("1.2.3+build.1", "1.2.3+build.99") == 0


@pytest.mark.parametrize("value", ["1.0", "01.0.0", "1.0.0-01", "1.0.0-"])
def test_invalid_semver_is_rejected(value):
    with pytest.raises(ValueError):
        compare_versions(value, "1.0.0")
