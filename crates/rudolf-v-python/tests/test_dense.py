"""StereoDepth and StereoRectification: the bindings to rudolf_v::sgbm,
rudolf_v::stereo_net and rudolf_v::rectify.

The matchers' accuracy is tested in Rudolf-V; these check what the bindings add:
the rectified camera from numbers, depth as Z (not range along the ray), the
ZED factory file, arrays in and out, and errors instead of crashes. The
TensorRT test runs only where RUDOLF_V_TRT_ENGINE and RUDOLF_V_TRT_LIB name an
engine and Rudolf-V's librv_trt.so (RUDOLF_V_TRT_PLUGINS: plugin libraries,
colon-separated).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from rudolf_v import StereoDepth, StereoRectification

W, H = 640, 360
FX, CX, CY, BASELINE = 400.0, 320.0, 180.0, 0.12
Z, DISPARITY = 2.0, 24  # FX * BASELINE / Z


def plane_pair(w=W, h=H, disparity=DISPARITY, seed=0):
    """A fronto-parallel textured plane: what the left camera sees at x, the right
    one sees at x - disparity. Random 16 px blocks, softened, so the texture
    survives 8x downsampling."""
    rng = np.random.default_rng(seed)
    big = np.kron(rng.random((h // 16 + 2, (w + disparity) // 16 + 2)), np.ones((16, 16)))
    for axis in (0, 1):
        c = np.cumsum(big, axis=axis)
        big = (np.roll(c, -2, axis=axis) - np.roll(c, 3, axis=axis)) / 5.0
    big = big[8 : 8 + h, 8 : 8 + w + disparity]
    big = ((big - big.min()) / (big.max() - big.min()) * 200 + 30).astype(np.uint8)
    return np.ascontiguousarray(big[:, :w]), np.ascontiguousarray(big[:, disparity:])


def sgbm(gpu=False, **kw):
    return StereoDepth.sgbm(FX, FX, CX, CY, BASELINE, W, H, gpu=gpu, **kw)


def check_plane(z):
    """Z of the plane everywhere it has depth: no trend across the image, as
    there would be with range along the ray (up to 28 % more at the corners)."""
    assert z.dtype == np.float32 and z.shape == (H // 8, W // 8)
    have = np.isfinite(z)
    assert have.mean() > 0.6
    assert abs(np.median(z[have]) - Z) < 0.02 * Z
    for part in (z[:, 30:], z[:, :30], z[:20], z[25:]):  # right, left, top, bottom
        assert abs(np.nanmedian(part) - Z) < 0.02 * Z


def test_sgbm_depth_is_z():
    left, right = plane_pair()
    depth = sgbm()
    assert (depth.kind, depth.level) == ("sgbm", 3)
    z = depth.compute(left, right)
    check_plane(z)
    var = depth.variance()
    assert var.shape == z.shape and np.isfinite(var[np.isfinite(z)]).all()
    assert depth.confidence() is None


@pytest.fixture(scope="module")
def on_gpu():
    try:
        return sgbm(gpu=True)
    except RuntimeError as e:
        pytest.skip(f"no Vulkan GPU: {e}")


def test_gpu_sgbm_gives_the_cpus_depth(on_gpu):
    left, right = plane_pair()
    assert on_gpu.kind == "sgbm-gpu" and "sgbm-gpu" in repr(on_gpu)
    z = on_gpu.compute(left, right)
    check_plane(z)
    cpu = sgbm().compute(left, right)
    same = (np.isnan(z) & np.isnan(cpu)) | np.isclose(z, cpu, rtol=1e-5)
    assert same.mean() > 0.99


def test_sgbm_errors():
    left, right = plane_pair()
    depth = sgbm()
    with pytest.raises(ValueError, match=r"\(360, 640\)"):
        depth.compute(left[:, 1:], right)
    with pytest.raises(ValueError, match="baseline"):
        StereoDepth.sgbm(FX, FX, CX, CY, -0.12, W, H, gpu=False)
    with pytest.raises(ValueError, match="too small"):  # 5x2 pixels at level 7
        sgbm(level=7)
    with pytest.raises(TypeError):
        depth.compute(left.astype(np.float32), right)


def test_sgbm_settings():
    left, right = plane_pair()
    default = sgbm().compute(left, right)
    spelled_out = sgbm(block_size=5, p1=200, p2=800, pre_filter_cap=63, uniqueness=10,
                       disp12_max_diff=1, speckle_size=400, disparity_sigma=0.5)
    assert np.array_equal(spelled_out.compute(left, right), default, equal_nan=True)

    # A patch with nothing to match (each eye its own random blocks): stricter
    # checks keep less of what is matched there, looser ones more.
    rng = np.random.default_rng(5)
    unmatched = [img.copy() for img in (left, right)]
    for img in unmatched:
        img[96:264, 192:448] = np.kron(rng.integers(30, 230, (21, 32)), np.ones((8, 8)))
    def kept(**kw):
        return np.isfinite(sgbm(**kw).compute(*unmatched)[12:33, 24:56]).sum()
    n = kept()
    assert kept(uniqueness=30) < n and kept(speckle_size=3200) < n
    assert kept(uniqueness=30, speckle_size=3200) < min(kept(uniqueness=30), kept(speckle_size=3200))
    assert kept(disp12_max_diff=None) > n and kept(speckle_size=0) > n

    # The variance is (sigma / disparity)².
    wide = sgbm(disparity_sigma=1.0)
    assert np.array_equal(wide.compute(left, right), default, equal_nan=True)
    have = np.isfinite(default)
    np.testing.assert_allclose(wide.variance()[have], 4 * spelled_out.variance()[have], rtol=1e-6)

    with pytest.raises(ValueError, match="odd"):
        sgbm(block_size=4)
    with pytest.raises(ValueError, match="4 or 8"):
        sgbm(paths=6)
    with pytest.raises(ValueError, match="disparity_sigma"):
        sgbm(disparity_sigma=0.0)


def test_gpu_sgbm_settings(on_gpu):
    left, right = plane_pair()
    kw = dict(block_size=3, p2=500, uniqueness=20, speckle_size=1600)
    z = sgbm(gpu=True, **kw).compute(left, right)
    cpu = sgbm(**kw).compute(left, right)
    same = (np.isnan(z) & np.isnan(cpu)) | np.isclose(z, cpu, rtol=1e-5)
    assert same.mean() > 0.99
    with pytest.raises(ValueError, match="16-bit"):  # 11x11 blocks overflow the GPU's path sums
        sgbm(gpu=True, block_size=11)


# The factory file of the lab Orin's ZED 2i (S/N 36835247), HD only, and the
# SDK's rectified camera for it.
ZED_CONF = """[LEFT_CAM_HD]
fx=958.245
fy=958.715
cx=651.625
cy=353.5555
k1=-0.0720847
k2=-0.00535637
p1=0.00042353
p2=2.62033e-05
k3=-0.0735599

[RIGHT_CAM_HD]
fx=956.08
fy=956.605
cx=653.05
cy=359.875
k1=-0.0701151
k2=-0.0175723
p1=0.000313971
p2=4.32652e-05
k3=-0.0535535

[LEFT_DISTO]
k1=-4.26217
k2=46.5049
k3=-23.2569
k4=-4.16774
k5=46.0752
k6=-19.7633
p1=0.00042056
p2=9.13147e-06

[RIGHT_DISTO]
k1=-5.1226
k2=47.702
k3=-22.8344
k4=-5.03039
k5=47.2158
k6=-19.2661
p1=0.00032012
p2=2.33182e-05

[STEREO]
Baseline=120.042
TY=-0.252054
TZ=0.432304
CV_HD=0.00391325
RX_HD=0.00305798
RZ_HD=0.00347634
"""
SDK_RECT = (951.3826904296875, 658.940185546875, 356.7108459472656)


@pytest.fixture
def zed_conf(tmp_path):
    path = tmp_path / "SN36835247.conf"
    path.write_text(ZED_CONF)
    return str(path)


def test_rectification_from_the_zed_file(zed_conf):
    rect = StereoRectification.from_zed_conf(zed_conf, *SDK_RECT)
    assert (rect.width, rect.height) == (1280, 720)
    assert rect.fx == rect.fy == SDK_RECT[0] and (rect.cx, rect.cy) == SDK_RECT[1:]
    assert abs(rect.baseline - 0.120043054) < 1e-6  # the SDK's rectified baseline
    assert "baseline=0.120043" in repr(rect)

    raw = plane_pair(1280, 720)
    left, right = rect.rectify(*raw)
    for img in (left, right):
        assert img.dtype == np.uint8 and img.shape == (720, 1280)
    flat = np.full((720, 1280), 100, np.uint8)
    left, _ = rect.rectify(flat, flat)
    assert (left[100:-100, 100:-100] == 100).all()  # bilinear of a constant

    # The rectified camera feeds StereoDepth as is.
    depth = StereoDepth.sgbm(rect.fx, rect.fy, rect.cx, rect.cy, rect.baseline, rect.width, rect.height, gpu=False)
    assert depth.compute(*rect.rectify(*raw)).shape == (90, 160)


def test_gpu_rectification_gives_the_cpus_images(zed_conf):
    try:
        on_gpu = StereoRectification.from_zed_conf(zed_conf, *SDK_RECT, gpu=True)
    except RuntimeError as e:
        pytest.skip(f"no Vulkan GPU: {e}")
    cpu = StereoRectification.from_zed_conf(zed_conf, *SDK_RECT)
    assert on_gpu.gpu and not cpu.gpu and "gpu=True" in repr(on_gpu)
    for seed in (0, 1):  # twice: the second pair reuses every GPU buffer
        raw = plane_pair(1280, 720, seed=seed)
        for a, b in zip(on_gpu.rectify(*raw), cpu.rectify(*raw)):
            assert a.dtype == np.uint8 and a.shape == (720, 1280)
            d = np.abs(a.astype(int) - b)
            assert d.max() <= 1 and (d > 0).mean() < 1e-4  # fused multiply-adds, on a few pixels


def test_radtan_and_kalibr_are_other_ways_in(zed_conf, tmp_path):
    raw = plane_pair(1280, 720)
    rational = StereoRectification.from_zed_conf(zed_conf, *SDK_RECT).rectify(*raw)[0]
    radtan = StereoRectification.from_zed_conf(zed_conf, *SDK_RECT, distortion="radtan").rectify(*raw)[0]
    assert 0 < np.abs(rational[200:-200, 300:-300].astype(int) - radtan[200:-200, 300:-300]).mean() < 10

    # The SDK's own export of the same calibration, as Kalibr's camchain.yaml.
    camchain = tmp_path / "camchain.yaml"
    camchain.write_text("""cam0:
  camera_model: pinhole
  intrinsics: [958.2449951171875, 958.7150268554688, 651.625, 353.5555114746094]
  distortion_model: rational_polynomial
  distortion_coeffs: [-4.26216983795166, 46.50490188598633, 0.00042056001257151365,
    9.131469596468378e-06, -23.256900787353516, -4.1677398681640625, 46.075199127197266,
    -19.7632999420166]
  resolution: [1280, 720]
cam1:
  camera_model: pinhole
  intrinsics: [956.0800170898438, 956.60498046875, 653.0499877929688, 359.875]
  distortion_model: rational_polynomial
  distortion_coeffs: [-5.122600078582764, 47.70199966430664, 0.00032011998700909317,
    2.3318199964705855e-05, -22.834400177001953, -5.030389785766602, 47.21580123901367,
    -19.26609992980957]
  resolution: [1280, 720]
  T_cn_cnm1:
  - [0.999986291, -0.003470335, 0.003918541, -0.120042004]
  - [0.003482302, 0.999989271, -0.003051159, -0.000252054]
  - [-0.003907911, 0.003064763, 0.999987662, 0.000432304]
  - [0.0, 0.0, 0.0, 1.0]
""")
    kalibr = StereoRectification.from_kalibr(str(camchain), *SDK_RECT).rectify(*raw)
    zed = StereoRectification.from_zed_conf(zed_conf, *SDK_RECT).rectify(*raw)
    for a, b in zip(kalibr, zed):
        assert np.abs(a.astype(int) - b).max() <= 1  # float32 rounding in the SDK's export


def test_rectification_errors(zed_conf):
    with pytest.raises(ValueError, match="FHD"):
        StereoRectification.from_zed_conf(zed_conf, *SDK_RECT, resolution="FHD")
    with pytest.raises(ValueError, match="resolution"):
        StereoRectification.from_zed_conf(zed_conf, *SDK_RECT, resolution="4K")
    with pytest.raises(ValueError, match="distortion"):
        StereoRectification.from_zed_conf(zed_conf, *SDK_RECT, distortion="fisheye")
    with pytest.raises(ValueError, match="Failed to read"):
        StereoRectification.from_zed_conf(zed_conf + ".missing", *SDK_RECT)
    rect = StereoRectification.from_zed_conf(zed_conf, *SDK_RECT)
    with pytest.raises(ValueError, match=r"\(720, 1280\)"):
        rect.rectify(*plane_pair())


@pytest.mark.skipif(
    not (os.environ.get("RUDOLF_V_TRT_ENGINE") and os.environ.get("RUDOLF_V_TRT_LIB")),
    reason="no TensorRT engine (RUDOLF_V_TRT_ENGINE, RUDOLF_V_TRT_LIB)",
)
def test_tensorrt_network():
    plugins = [p for p in os.environ.get("RUDOLF_V_TRT_PLUGINS", "").split(":") if p]
    depth = StereoDepth.tensorrt(
        os.environ["RUDOLF_V_TRT_ENGINE"], FX, FX, CX, CY, BASELINE, W, H,
        library=os.environ["RUDOLF_V_TRT_LIB"], plugins=plugins,
    )
    assert depth.kind in ("light-ess", "las2")
    z = depth.compute(*plane_pair())
    assert z.shape == (H // 8, W // 8) and np.isfinite(z).mean() > 0.5
    assert abs(np.nanmedian(z) - Z) < 0.15 * Z
    conf = depth.confidence()
    if depth.kind == "light-ess":
        assert conf.shape == z.shape and ((conf >= 0) & (conf <= 1)).all()
    else:
        assert conf is None
    with pytest.raises(RuntimeError, match="engine|lib|No such"):
        StereoDepth.tensorrt("/nonexistent.engine", FX, FX, CX, CY, BASELINE, W, H, library=os.environ["RUDOLF_V_TRT_LIB"])
