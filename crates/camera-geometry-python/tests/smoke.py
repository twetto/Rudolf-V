from math import isfinite

from camera_geometry import EquidistantFisheye


camera = EquidistantFisheye(
    [190.97847715128717, 190.9733070521226, 254.93170605935475, 256.8974428996504],
    [0.0034823894022493434, 0.0007150348452162257, -0.0020532361418706202, 0.00020293673591811182],
    [512, 512],
)

assert camera.image_size == [512, 512]
for pixel in ([254.93170605935475, 256.8974428996504], [1.0, 1.0], [510.0, 510.0]):
    bearing = camera.unproject(pixel)
    assert bearing is not None
    projected = camera.project(bearing)
    assert projected is not None
    assert max(abs(a - b) for a, b in zip(projected, pixel)) < 1.0e-8
    jacobian = camera.unproject_tangent_jacobian(pixel)
    assert jacobian is not None
    assert all(isfinite(value) for row in jacobian for value in row)

assert camera.project([0.0, 0.0, 0.0]) is None
print("camera-geometry wheel smoke test passed")
