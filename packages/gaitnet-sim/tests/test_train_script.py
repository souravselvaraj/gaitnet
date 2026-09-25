"""The train entry point's own arguments are consumed before Isaac Lab parses the rest."""

from __future__ import annotations

import sys
import types

import pytest
import torch

from gaitnet_sim.scripts import train


@pytest.fixture
def captured(monkeypatch):
    calls = []
    module = types.ModuleType("isaaclab_rl.entrypoints.backends.train_rsl_rl")
    module.run = calls.append
    monkeypatch.setitem(sys.modules, "isaaclab_rl.entrypoints.backends.train_rsl_rl", module)
    monkeypatch.setattr(train, "_diff_this_repo_only", lambda: None)
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
