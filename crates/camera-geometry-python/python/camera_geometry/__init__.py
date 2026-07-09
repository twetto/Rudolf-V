"""Calibrated camera projection backed by the Rust camera-geometry crate."""

from ._native import EquidistantFisheye

__all__ = ["EquidistantFisheye"]
