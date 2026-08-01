import pytest

from bmodel.config import reset_defaults


@pytest.fixture(autouse=True)
def _reset_live_models():
    yield
    reset_defaults()
