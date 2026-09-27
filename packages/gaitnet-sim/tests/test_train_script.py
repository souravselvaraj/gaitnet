"""The train entry point's own arguments are consumed before Isaac Lab parses the rest."""

from __future__ import annotations

import sys
import types

import pytest
import torch

from gaitnet_sim.scripts import train


class CallLog(list):
    carried: list


@pytest.fixture
def captured(monkeypatch):
    calls = CallLog()
    module = types.ModuleType("isaaclab_rl.entrypoints.backends.train_rsl_rl")
    module.run = calls.append
    monkeypatch.setitem(sys.modules, "isaaclab_rl.entrypoints.backends.train_rsl_rl", module)
    monkeypatch.setattr(train, "_diff_this_repo_only", lambda: None)
    carried = []
    monkeypatch.setattr(train, "_carry_curriculum", lambda restore, initial: carried.append((restore, initial)))
    calls.carried = carried
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    return calls


def test_tf32_is_consumed_and_enabled(captured):
    train.main(["--task", "GaitNet-Holes", "--tf32", "presets=privileged"])
    assert captured == [["--external_callback", "gaitnet_sim.tasks.register", "--task", "GaitNet-Holes", "presets=privileged"]]
    assert torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32


def test_tf32_is_off_unless_asked(captured):
    train.main(["--task", "GaitNet-Holes"])
    assert captured[0][-2:] == ["--task", "GaitNet-Holes"]
    assert not torch.backends.cuda.matmul.allow_tf32


def test_curriculum_flags_are_consumed(captured):
    train.main(["--task", "GaitNet-Holes", "--init_terrain_levels", "3:11", "--checkpoint", "m.pt"])
    assert captured[0][-4:] == ["--task", "GaitNet-Holes", "--checkpoint", "m.pt"]
    assert captured.carried == [(True, (3, 11))]
    train.main(["--task", "GaitNet-Holes", "--reset_curriculum"])
    assert captured[1][-2:] == ["--task", "GaitNet-Holes"]
    assert captured.carried[1] == (False, None)
