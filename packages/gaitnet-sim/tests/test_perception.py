"""The front camera's elevation map: what it sees, what it remembers, how wrong it is. No
simulator."""

from __future__ import annotations

import math

import pytest
import torch

pytest.importorskip("isaaclab")

from gaitnet_sim.env.perception import FrontCameraCfg, FrontCameraMap  # noqa: E402

HEIGHT = 0.27  # base above the ground (m)


def level(yaw: float = 0.0, n: int = 1) -> torch.Tensor:
    return torch.tensor([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]).expand(n, 4).clone()


def fresh_map(n: int = 1) -> FrontCameraMap:
    camera = FrontCameraMap(FrontCameraCfg(), n, "cpu")
    camera.reset(torch.arange(n), torch.zeros(n, 2))
    return camera


def at(x: float, n: int = 1) -> torch.Tensor:
    return torch.tensor([x, 0.0, HEIGHT]).expand(n, 3).clone()


def known_at(camera: FrontCameraMap, x: float, y: float = 0.0) -> bool:
    known, _ = camera.lookup(torch.tensor([[[x, y]]]))
    return bool(known[0, 0])


def test_reset_knows_only_the_spawn_platform():
    camera = fresh_map()
    assert known_at(camera, 0.0) and known_at(camera, 0.55, -0.55)
    assert not known_at(camera, 0.7) and not known_at(camera, -0.7)


def test_the_camera_sees_ahead_not_below_or_behind():
    camera = fresh_map()
    camera.update(at(0.0), level(), torch.zeros(1))
    assert known_at(camera, 1.0) and known_at(camera, 2.0)
    assert not known_at(camera, 0.7, 0.9)  # beside the front feet, below the view
    assert not known_at(camera, -1.0) and not known_at(camera, 0.0, 1.5)
    index, seen, distance = camera.visible(at(0.0), level(), torch.zeros(1))
    xs = (index[..., 0].float() + 0.5) * camera.cfg.map_resolution
    # nearest ground in view: camera 0.37 m ahead and 0.28 m up, bottom of view 51 deg down
    assert xs[seen].min() == pytest.approx(0.36 + 0.28 / math.tan(0.38 + math.radians(29)), abs=0.05)
    assert distance[seen].max() <= 3.0


def test_the_map_rolls_with_the_robot():
    camera = fresh_map()
    for step in range(101):  # 4 m at 0.04 m per tick, looking ahead all the way
        camera.update(at(step * 0.04), level(), torch.zeros(1))
    # the ground under and ahead of the robot, 4 m from its spawn, is known
    assert known_at(camera, 4.2) and known_at(camera, 5.5)
    # more than a window (5 m) behind is forgotten, though it was seen
    assert known_at(camera, 1.8) and not known_at(camera, -0.3)


def test_turning_points_the_camera():
    camera = fresh_map()
    camera.update(at(0.0), level(math.pi / 2), torch.zeros(1))
    assert known_at(camera, 0.0, 1.5) and not known_at(camera, 1.5, 0.0)


def test_depth_error_grows_with_range_and_fusion_shrinks_it():
    torch.manual_seed(0)
    n = 2000
    camera = fresh_map(n)
    near = torch.tensor([[1.0, 0.0]]).expand(n, 2).reshape(n, 1, 2)
    far = torch.tensor([[2.5, 0.0]]).expand(n, 2).reshape(n, 1, 2)
    camera.update(at(0.0, n), level(n=n), torch.zeros(n))
    near_std = camera.lookup(near)[1].std()
    far_std = camera.lookup(far)[1].std()
    assert far_std > 3 * near_std
    for _ in range(9):
        camera.update(at(0.0, n), level(n=n), torch.zeros(n))
    # ten fused measurements: about 1/sqrt(10) of one
    assert camera.lookup(far)[1].std() == pytest.approx(far_std / math.sqrt(10), rel=0.2)


def test_reset_forgets_one_robot_only():
    camera = fresh_map(2)
    camera.update(at(0.0, 2), level(n=2), torch.zeros(2))
    camera.reset(torch.tensor([1]), torch.tensor([[5.0, 5.0]]))
    known, _ = camera.lookup(torch.tensor([[[1.5, 0.0]], [[1.5, 0.0]]]))
    assert known[0, 0] and not known[1, 0]
    known, _ = camera.lookup(torch.tensor([[[5.0, 5.0]], [[5.0, 5.0]]]))
    assert known[1, 0]
