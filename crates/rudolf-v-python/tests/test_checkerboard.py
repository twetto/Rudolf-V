"""CheckerboardDetector: the binding to rudolf_v::checkerboard.

Boards are rendered here (4x4 supersampled, turned by any angle), so every test
knows the true inner corners. The detector's own accuracy is tested in Rudolf-V;
these check what the binding adds: arrays in and out, the region, the thread
pool, strided input, the GPU option, and errors instead of crashes.
"""

from __future__ import annotations

import numpy as np
import pytest

from rudolf_v import CheckerboardDetector

COLS, ROWS = 9, 6
DARK, LIGHT, BACKGROUND = 40.0, 210.0, 128.0


def render(w, h, cx, cy, square, deg=0.0, ss=4, seed=0):
    """A (COLS+1) x (ROWS+1)-square board with a one-square white margin, centred
    on (cx, cy) and turned by `deg`. Returns the image and the true inner corners
    (row-major)."""
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    yy, xx = np.mgrid[0 : h * ss, 0 : w * ss]
    px, py = (xx + 0.5) / ss - 0.5 - cx, (yy + 0.5) / ss - 0.5 - cy
    u = (c * px + s * py) / square + (COLS + 1) / 2
    v = (-s * px + c * py) / square + (ROWS + 1) / 2
    board = (u >= 0) & (u < COLS + 1) & (v >= 0) & (v < ROWS + 1)
    margin = (u >= -1) & (u < COLS + 2) & (v >= -1) & (v < ROWS + 2)
    dark = (np.floor(u) + np.floor(v)) % 2 == 0
    img = np.where(board, np.where(dark, DARK, LIGHT), np.where(margin, LIGHT, BACKGROUND))
    img = img.reshape(h, ss, w, ss).mean(axis=(1, 3))
    img += np.random.default_rng(seed).integers(-4, 5, img.shape)
    i, j = np.meshgrid(np.arange(COLS) + 1 - (COLS + 1) / 2, np.arange(ROWS) + 1 - (ROWS + 1) / 2)
    bx, by = i.ravel() * square, j.ravel() * square
    truth = np.stack([cx + c * bx - s * by, cy + s * bx + c * by], axis=1)
    return np.clip(img, 0, 255).astype(np.uint8), truth


def max_error(found, truth):
    """Largest corner error, in order or with the 180-degree flip (the same board)."""
    return min(np.abs(found - truth).max(), np.abs(found[::-1] - truth).max())


def test_finds_the_board_subpixel():
    img, truth = render(400, 300, 200.3, 150.6, 24.0)
    found = CheckerboardDetector().detect(img)
    assert found.dtype == np.float32 and found.shape == (COLS * ROWS, 2)
    assert max_error(found, truth) < 0.15
    # row-major, rows along +x, rows ordered +y
    assert found[1, 0] > found[0, 0] and found[COLS, 1] > found[0, 1]


@pytest.mark.parametrize("deg", [30.0, 90.0, -90.0, 145.0])
def test_turned_boards_are_never_mirrored(deg):
    img, truth = render(400, 400, 200.3, 200.6, 22.0, deg=deg)
    found = CheckerboardDetector().detect(img)
    assert found is not None and max_error(found, truth) < 0.3


def test_no_board_is_none():
    rng = np.random.default_rng(1)
    assert CheckerboardDetector().detect(rng.integers(0, 256, (240, 320), dtype=np.uint8)) is None
    assert CheckerboardDetector().detect(np.full((240, 320), 128, np.uint8)) is None
    assert CheckerboardDetector().detect(np.zeros((5, 5), np.uint8)) is None


def test_other_pattern_sizes():
    img, _ = render(400, 300, 200.3, 150.6, 24.0)
    assert CheckerboardDetector(cols=7, rows=6).detect(img) is None  # a 9x6 board is not a 7x6 one
    d = CheckerboardDetector(cols=6, rows=9)  # either orientation
    assert d.cols == 6 and d.rows == 9 and d.detect(img) is not None


def test_region_in_whole_image_pixels_and_clipped():
    img, truth = render(480, 360, 300.3, 200.6, 20.0)
    d = CheckerboardDetector()
    found = d.detect_region(img, 180, 110, 260, 190)
    assert found is not None and max_error(found, truth) < 0.15
    # reaching past the image is clipped, not an error
    assert max_error(d.detect_region(img, 150, -50, 1000, 1000), truth) < 0.15
    assert d.detect_region(img, 0, 0, 150, 360) is None  # misses the board
    assert d.detect_region(img, 500, 400, 50, 50) is None  # outside the image
    assert d.detect_region(img, 100, 100, 0, 50) is None  # empty


def test_strided_input_is_copied():
    img, truth = render(440, 340, 220.3, 170.6, 24.0)
    view = img[20:-20, 20:-20]  # rows not contiguous
    assert not view.flags["C_CONTIGUOUS"]
    found = CheckerboardDetector().detect(view)
    assert max_error(found, truth - 20) < 0.15


def test_own_pool_gives_the_same_corners():
    img, _ = render(400, 300, 200.3, 150.6, 24.0, deg=20.0)
    a = CheckerboardDetector().detect(img)
    b = CheckerboardDetector(threads=1).detect(img)
    c = CheckerboardDetector(threads=3).detect(img)
    assert np.array_equal(a, b) and np.array_equal(a, c)


def test_config_overrides_reach_the_detector():
    img, _ = render(400, 300, 200.3, 150.6, 24.0)
    d = CheckerboardDetector(min_cell_contrast=500.0)  # no board has that much contrast
    assert "min_cell_contrast=500" in repr(d)
    assert d.detect(img) is None


def test_bad_input_raises():
    with pytest.raises(ValueError):
        CheckerboardDetector(cols=1, rows=6)
    with pytest.raises(ValueError):
        CheckerboardDetector(threads=0)
    d = CheckerboardDetector()
    with pytest.raises(TypeError):
        d.detect(np.zeros((100, 100), np.float32))
    with pytest.raises(TypeError):
        d.detect(np.zeros((100, 100, 3), np.uint8))


@pytest.fixture(scope="module")
def on_gpu():
    try:
        return CheckerboardDetector(gpu=True)
    except RuntimeError as e:
        pytest.skip(f"no Vulkan GPU: {e}")


def test_gpu_gives_the_same_corners(on_gpu):
    cpu = CheckerboardDetector()
    assert on_gpu.gpu and not cpu.gpu and "gpu=True" in repr(on_gpu)
    rng = np.random.default_rng(2)
    images = [render(400, 300, 200.3, 150.6, 24.0)[0]]
    images += [render(400, 400, 200.3, 200.6, 22.0, deg=deg)[0] for deg in (30.0, 145.0)]
    images += [render(1280, 720, 640.4, 360.2, 60.0)[0]]  # big squares: the half size
    images += [rng.integers(0, 256, (240, 320), dtype=np.uint8), np.zeros((5, 5), np.uint8)]
    for img in images:
        a, b = cpu.detect(img), on_gpu.detect(img)
        assert (a is None and b is None) or np.array_equal(a, b)


def test_gpu_windows_of_changing_size(on_gpu):
    """As the mission looks: windows around the board, a different size each time."""
    img, truth = render(640, 480, 320.3, 240.6, 20.0)
    cpu = CheckerboardDetector()
    for x, y, w, h in [(150, 110, 340, 260), (170, 130, 301, 223), (0, 0, 640, 480), (140, 100, 361, 281)]:
        a = cpu.detect_region(img, x, y, w, h)
        b = on_gpu.detect_region(img, x, y, w, h)
        assert np.array_equal(a, b) and max_error(b, truth) < 0.15
        view = img[y : y + h, x : x + w]  # a strided crop, as the mission passes it
        assert np.array_equal(cpu.detect(view), on_gpu.detect(view))
