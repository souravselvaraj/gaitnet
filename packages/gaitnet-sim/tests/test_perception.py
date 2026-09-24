"""The front camera's elevation map: what it sees, what it remembers, how wrong it is. No
simulator."""

from __future__ import annotations

import math

import pytest
import torch

pytest.importorskip("isaaclab")

from gaitnet_sim.env.perception import FrontCameraCfg, FrontCameraMap  # noqa: E402

GROUND = 0.0
HEIGHT = 0.27  # base above the ground (m)


def level(yaw: float = 0.0, n: int = 1) -> torch.Tensor:
    return torch.tensor([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]).expand(n, 4).clone()


def fresh_map(cfg: FrontCameraCfg | None = None, n: int = 1) -> FrontCameraMap:
    camera = FrontCameraMap(cfg or FrontCameraCfg(), n, "cpu")
    camera.reset(torch.arange(n), torch.zeros(n, 2))
    return camera


def known_at(camera: FrontCameraMap, x: float, y: float = 0.0) -> bool:
    known, _ = camera.lookup(torch.tensor([[[x, y]]]))
    return bool(known[0, 0])


def test_reset_knows_only_the_spawn_platform():
    camera = fresh_map()
    assert known_at(camera, 0.0) and known_at(camera, 0.45, -0.45)
    assert not known_at(camera, 0.7) and not known_at(camera, -0.7)
    assert not known_at(camera, 10.0)  # outside the map


def test_the_camera_sees_ahead_not_below_or_behind():
    camera = fresh_map()
    camera.update(torch.tensor([[0.0, 0.0, HEIGHT]]), level(), torch.tensor([GROUND]))
    assert known_at(camera, 1.0) and known_at(camera, 2.0)
    # the front feet's ground is below the view (only the reset platform knows it)
    assert not known_at(camera, 0.7, 0.9)
    assert not known_at(camera, -1.0) and not known_at(camera, 0.0, 1.5)
    seen, distance = camera.visible(torch.tensor([[0.0, 0.0, HEIGHT]]), level(), torch.tensor([GROUND]))
    xs = camera._offsets[..., 0][seen[0]]
    # nearest ground in view: camera 0.37 m ahead and 0.28 m up, bottom of view 51 deg down
    assert xs.min() == pytest.approx(0.36 + 0.28 / math.tan(0.38 + math.radians(29)), abs=0.05)
    assert distance[seen].max() <= 3.0


def test_ground_seen_ahead_is_remembered_after_walking_onto_it():
    camera = fresh_map()
    camera.update(torch.tensor([[0.0, 0.0, HEIGHT]]), level(), torch.tensor([GROUND]))
    assert known_at(camera, 1.2)
    # the robot has walked 1 m on: 1.2 m is now under its front feet, below the view
    camera.update(torch.tensor([[1.0, 0.0, HEIGHT]]), level(), torch.tensor([GROUND]))
    assert known_at(camera, 1.2)
    seen, _ = camera.visible(torch.tensor([[1.0, 0.0, HEIGHT]]), level(), torch.tensor([GROUND]))
    index = int((1.2 + 3.0) / 0.03)
    assert not seen[0, index, 100]


def test_turning_points_the_camera():
    camera = fresh_map()
    camera.update(torch.tensor([[0.0, 0.0, HEIGHT]]), level(math.pi / 2), torch.tensor([GROUND]))
    assert known_at(camera, 0.0, 1.5) and not known_at(camera, 1.5, 0.0)


def test_depth_error_grows_with_range_and_fusion_shrinks_it():
    torch.manual_seed(0)
    camera = fresh_map(n=2000)
    pos = torch.tensor([0.0, 0.0, HEIGHT]).expand(2000, 3)
    camera.update(pos, level(n=2000), torch.zeros(2000))
    near = int((1.0 + 3.0) / 0.03)
    far = int((2.5 + 3.0) / 0.03)
    near_std = camera.error[:, near, 100].std()
    far_std = camera.error[:, far, 100].std()
    assert far_std > 3 * near_std
    for _ in range(9):
        camera.update(pos, level(n=2000), torch.zeros(2000))
    # ten fused measurements: about 1/sqrt(10) of one
    assert camera.error[:, far, 100].std() == pytest.approx(far_std / math.sqrt(10), rel=0.2)


def test_reset_forgets_one_robot_only():
    camera = fresh_map(n=2)
    camera.update(torch.tensor([[0.0, 0.0, HEIGHT]] * 2), level(n=2), torch.zeros(2))
    camera.reset(torch.tensor([1]), torch.tensor([[5.0, 5.0]]))
    known, _ = camera.lookup(torch.tensor([[[1.5, 0.0]], [[6.5, 5.0]]]))
    assert known[0, 0] and not known[1, 0]
