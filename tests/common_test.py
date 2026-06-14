import pytest
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common import dec_width, Color, get_color


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
def test_dec_width(n: int, w: int):
    assert dec_width(n) == w

@pytest.mark.parametrize('rgb, color', [
    ((80, 35, 20), Color.RED),
    ((180, 135, 120), Color.RED),
    ((255, 200, 210), Color.RED),
    ((255, 42, 100), Color.RED),
    ((255, 255, 255), Color.BLACK),
    ((0, 0, 0), Color.BLACK),
    ((120, 125, 115), Color.BLACK),
    ((165, 155, 145), Color.BLACK),
])
def test_color(rgb: tuple, color: Color):
    assert get_color(*rgb) == color
