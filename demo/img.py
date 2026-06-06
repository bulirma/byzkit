import numpy as np


def symmetric_pad(array: np.ndarray, axis: int, size: int, value):
    total = size - array.shape[axis]
    before = total // 2
    after = total - before
    pad_width = [(0, 0)] * array.ndim
    pad_width[axis] = (before, after)
    return np.pad(array, pad_width, mode='constant', constant_values=value)

def bresenham_filled_circle(cx: int, cy: int, d: int):
    r = d / 2
    if d % 2 == 0:
        cx_sub = cx + 0.5
        cy_sub = cy + 0.5
    else:
        cx_sub = cx
        cy_sub = cy

    r2 = r * r
    ir = int(r)

    for dy in range(-ir - 1, ir + 2):
        for dx in range(-ir - 1, ir + 2):
            px = cx + dx
            py = cy + dy
            dist_sq = (px - cx_sub) ** 2 + (py - cy_sub) ** 2
            if dist_sq <= r2:
                yield (px, py)

def bresenham_line(x0: int, y0: int, x1: int, y1: int):
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy

    while True:
        yield x0, y0
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
