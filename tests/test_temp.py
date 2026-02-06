"""
Simple temporary test to verify pytest setup.
"""


def test_simple_addition():
    """Test that basic addition works."""
    assert 1 + 1 == 2


def test_simple_string():
    """Test that basic string operations work."""
    assert "hello" + " world" == "hello world"


def test_always_passes():
    """Test that should always pass."""
    assert True
