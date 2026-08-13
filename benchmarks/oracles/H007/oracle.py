import pytest

from library.core import dependency_order


def test_dependency_order_contract():
    order = dependency_order({"api": ["db", "auth"], "auth": ["db"], "db": [], "docs": []})
    assert set(order) == {"api", "auth", "db", "docs"}
    assert order.index("db") < order.index("auth") < order.index("api")


def test_cycles_are_rejected():
    with pytest.raises(ValueError):
        dependency_order({"a": ["b"], "b": ["a"]})
