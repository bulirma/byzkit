import pytest
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common import dec_width


@pytest.mark.parametrize('n, w', [
    (0, 1),
    (1, 1),
    (9, 1),
    (10, 2),
    (42, 2),
    (100, 3),
    (999, 3),
    (1000, 4),
    (10000, 5),
    (65536, 5),
    (100000, 6)
])
def test_dec_width(n, w):
    assert dec_width(n) == w
