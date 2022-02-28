"""
Unit and regression test for the hesmatch package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import hesmatch


def test_hesmatch_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "hesmatch" in sys.modules
