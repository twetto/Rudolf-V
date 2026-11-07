"""Frontend, FrontendConfig and Stereo: the bindings to rudolf_v::frontend,
rudolf_v::stereo and rudolf_v::config.

The tracker's and the matcher's accuracy is tested in Rudolf-V; these check what
the bindings add: the config file, names instead of enums, arrays and dicts in
and out, and errors instead of crashes.
"""

from __future__ import annotations

import numpy as np
import pytest

from rudolf_v import Frontend, FrontendConfig, Stereo

W, H = 320, 240

# A whole config file, as ECHO-LI writes one: the bindings read their own
# sections and nothing else.
CONFIG = """
RudolfV:
  equaliseImageHistogram: true
  featureDist: 16.0
  featureSearchThreshold: 0.8
  fastThreshold: 20
  detector: shi_tomasi
  maxFeatures: 300
  maxLevel: 3
  lbpPolicy: HardReject
  enableRansac: true
Stereo:
  enabled: true
  pyramid_levels: 3
  max_iterations: 30
  histeq: global
eqf:
  maxFeatures: 40
"""


def texture(w, h, seed=0):
    """Random 6 px blocks, softened by a 3 px box blur: corners everywhere, and
    nothing repeats."""
    rng = np.random.default_rng(seed)
    img = np.kron(rng.random((h // 6 + 4, w // 6 + 4)), np.ones((6, 6)))
    for axis in (0, 1):
        c = np.cumsum(img, axis=axis)
        img = (np.roll(c, -1, axis=axis) - np.roll(c, 2, axis=axis)) / 3.0
    img = img[12 : 12 + h, 12 : 12 + w]
    img = (img - img.min()) / (img.max() - img.min())
    return (img * 220 + 20).astype(np.uint8)


@pytest.fixture
def config_file(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(CONFIG)
    return str(path)


def test_from_yaml_reads_the_rudolf_v_section(config_file):
    c = FrontendConfig.from_yaml(config_file)
    assert (c.max_features, c.cell_size, c.pyramid_levels, c.fast_threshold) == (300, 16, 3, 20)
    assert (c.detector, c.lbp_policy, c.histeq) == ("shi_tomasi", "hard", "global")
    assert c.enable_ransac and not c.klt_residual
    assert "detector=shi_tomasi" in repr(c)


def test_config_file_errors(tmp_path):
    def file(text):
        path = tmp_path / "c.yaml"
        path.write_text(text)
        return str(path)

    with pytest.raises(ValueError, match="no RudolfV section"):
        FrontendConfig.from_yaml(file("eqf: {}\n"))
    with pytest.raises(ValueError, match="maxFeature"):  # a typo is not ignored
        FrontendConfig.from_yaml(file(CONFIG.replace("maxFeatures: 300", "maxFeature: 300")))
    with pytest.raises(ValueError, match="RudolfV.histeq 'adaptive'"):
        FrontendConfig.from_yaml(file(CONFIG.replace("lbpPolicy", "histeq: adaptive\n  lbpPolicy")))
    with pytest.raises(OSError):
        FrontendConfig.from_yaml(str(tmp_path / "missing.yaml"))


def test_names_are_checked():
    for field, value in [("histeq", "adaptive"), ("detector", "orb"), ("lbp_policy", "maybe"), ("klt_warp", "affine")]:
        c = FrontendConfig()
        setattr(c, field, value)
        with pytest.raises(ValueError, match=field.split("_")[0]):
            Frontend(c, W, H)
    c = FrontendConfig(histeq="CLAHE", detector="Shi-Tomasi", lbp_policy="HardReject")
    Frontend(c, W, H)
    with pytest.raises(ValueError, match="distortion_model"):
        c.set_camera(300.0, 300.0, 160.0, 120.0, W, H, [0.0] * 4, "kannala")


def test_tracks_a_moving_image():
    big = texture(W + 16, H + 16)
    fe = Frontend(FrontendConfig(max_features=100, cell_size=32, histeq="none"), W, H)
    first, stats = fe.process(np.ascontiguousarray(big[8 : 8 + H, 8 : 8 + W]))
    assert len(first) > 20 and stats["new_detections"] == len(first)
    assert {"timing_ms", "histeq_ms", "pyramid_ms", "klt_ms", "ransac_ms", "detect_ms"} <= stats.keys()

    # the camera moves; the image moves by (-2, -1) px
    second, stats = fe.process(np.ascontiguousarray(big[9 : 9 + H, 10 : 10 + W]))
    before = {f["id"]: (f["x"], f["y"]) for f in first}
    moved = [(f["x"] - before[f["id"]][0], f["y"] - before[f["id"]][1]) for f in second if f["id"] in before]
    assert stats["tracked"] == len(moved) > 0.8 * len(first)
    assert np.allclose(np.median(moved, axis=0), (-2.0, -1.0), atol=0.05)
    assert len(fe.features()) == len(second) and len(fe.track_meta()) == len(second)

    gone = [f["id"] for f in second[:5]]
    assert fe.drop_tracks(gone) == 5
    assert not {f["id"] for f in fe.features()} & set(gone)
    fe.reset()
    assert fe.features() == []


def test_frontend_images():
    fe = Frontend(FrontendConfig(histeq="global"), W, H)
    assert fe.preprocessed_image() is None
    img = texture(W, H)
    fe.process(img)
    eq = fe.preprocessed_image()
    assert eq.dtype == np.uint8 and eq.shape == (H, W) and eq.max() == 255
    with pytest.raises(ValueError, match="doesn't match"):
        fe.process(np.zeros((H, W + 1), np.uint8))
    with pytest.raises(ValueError, match="4x4"):
        fe.set_pose_prior(np.eye(3))
    fe.set_pose_prior(np.eye(4))


def test_stereo_depth_of_a_plane(config_file):
    """cam1 is 0.1 m right of cam0; a plane 5 m away is 8 px of disparity."""
    fx, baseline, disparity = 400.0, 0.1, 8
    big = texture(W + disparity, H)
    left = np.ascontiguousarray(big[:, :W])
    right = np.ascontiguousarray(big[:, disparity:])  # a point at x in cam0 is at x - 8 in cam1
    # cam0's equalization must be the Stereo section's, or the patches differ
    fe = Frontend(FrontendConfig(max_features=100, cell_size=32, histeq="global"), W, H)
    fe.process(left)
    stereo = Stereo.from_pinhole(fx, fx, W / 2, H / 2, W, H, [-baseline, 0.0, 0.0], vio_config=config_file)
    assert abs(stereo.baseline_meters - baseline) < 1e-12

    matches = stereo.match_features(right, fe)
    assert len(matches) == len(fe.features())
    z = [m["point_cam0"][2] for m in matches if m["matched"]]
    assert len(z) > 0.6 * len(matches)
    assert abs(np.median(z) - fx * baseline / disparity) < 0.01

    priors = stereo.range_priors(right, fe)
    assert len(priors) == len(z)
    assert all(r > 4.9 and var >= 1e-4 for r, var in priors.values())
    with pytest.raises(ValueError, match="doesn't match"):
        stereo.match_features(right[:, 1:], fe)


def test_stereo_config_errors(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("Stereo:\n  histeq: clahe\n")
    with pytest.raises(ValueError, match="Stereo.histeq"):
        Stereo.from_pinhole(400.0, 400.0, 160.0, 120.0, W, H, [-0.1, 0.0, 0.0], vio_config=str(path))
    path.write_text("eqf: {}\n")  # no Stereo section: the defaults
    Stereo.from_pinhole(400.0, 400.0, 160.0, 120.0, W, H, [-0.1, 0.0, 0.0], vio_config=str(path))
