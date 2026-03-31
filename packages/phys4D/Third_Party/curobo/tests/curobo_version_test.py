










"""Unit tests for `curobo` package version."""


import curobo


def test_curobo_version():
    """Test `curobo` package version is set."""
    assert curobo.__version__ is not None
    assert curobo.__version__ != ""
