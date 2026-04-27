import pytest
from scgpt import __version__


@pytest.mark.skip(reason="Legacy version pin for 0.1.0 tutorials; package is now 0.2.5+")
def test_version():
    # legacy package name check
    assert __version__ == "0.1.0"
