import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common import dec_width

def test_dec_width():
    assert dec_width(0) == 1
    assert dec_width(1) == 1
    assert dec_width(9) == 1
    assert dec_width(10) == 2
    assert dec_width(42) == 2
    assert dec_width(100) == 3
    assert dec_width(999) == 3
    assert dec_width(1000) == 4
    assert dec_width(10000) == 5
    assert dec_width(65536) == 5
    assert dec_width(100000) == 6
