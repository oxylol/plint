import pytest

from sanity_testkit import context_with


@pytest.fixture()
def ctx():
    return context_with()
