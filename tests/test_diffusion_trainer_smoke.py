"""Smoke test for local diffuser training CLI path."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from planning.diffusion.config import DiffusionTrainingConfig
from planning.diffusion.core import PlannerStateNormalizer


def test_training_smoke(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from planning.diffusion.training.trainer import DiffusionTrainingPipeline

    dataset = np.random.RandomState(0).randn(24, 5, 3)
    dataset_path = tmp_path / "toy.npz"
    np.savez(dataset_path, observations=dataset)

    checkpoints = DiffusionTrainingPipeline(
        dataset=str(dataset_path),
        output_path=str(tmp_path / "logs"),
        horizon=5,
        state_dim=3,
        n_diffusion_steps=16,
        epochs=1,
        batch_size=24,
        n_hidden=16,
        train_value=False,
    ).run()

    assert checkpoints
    assert all((tmp_path / "logs").rglob("*.ckpt"))
    assert (tmp_path / "logs" / "diffusion.pt").exists()


def test_training_package_exports_pipeline() -> None:
    from planning.diffusion.training import DiffusionTrainingPipeline as PackagePipeline
    from planning.diffusion.training.trainer import DiffusionTrainingPipeline as ModulePipeline

    assert PackagePipeline is ModulePipeline


def test_hydra_cli_overrides() -> None:
    from hydra import compose, initialize

    with initialize(config_path="../config", version_base=None):
        cfg = compose(
            config_name="diffusion_3d_training",
            overrides=[
                "dataset=dummy.npz",
                "output_path=logs/custom_root",
                "+diffusion_max_epochs=2",
                "+value_patience=3",
                "+diffusion_min_delta=0.05",
                "checkpoint_every=3",
                "keep_last_checkpoints=2",
                "n_hidden=64",
                "device=cuda:0",
            ],
        )

    assert cfg.dataset == "dummy.npz"
    assert cfg.diffusion_max_epochs == 2
    assert cfg.value_patience == 3
    assert cfg.diffusion_min_delta == 0.05
    assert "tensorboard_log_dir" not in cfg
    assert cfg.checkpoint_every == 3
    assert cfg.keep_last_checkpoints == 2
    assert cfg.n_hidden == 64
    assert cfg.device == "cuda:0"

    pipeline_cfg = DiffusionTrainingConfig(**dict(cfg))
    assert (Path(pipeline_cfg.output_path) / "tensorboard") == Path("logs/custom_root/tensorboard")


def test_training_pipeline_uses_cpu_tensor_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("torch")
    from planning.diffusion.training import trainer

    dataset = np.random.RandomState(0).randn(16, 5, 3)
    dataset_path = tmp_path / "toy.npz"
    np.savez(dataset_path, observations=dataset)

    base_factory = trainer.TorchTensorFactory
    call_count = {"init": 0}

    class CpuOnlyFactory:
        def __init__(self, normalizer: object) -> None:
            call_count["init"] += 1
            self._inner = base_factory(cast(PlannerStateNormalizer, normalizer))

        def to_torch_tensors(self, trajectories: np.ndarray) -> tuple[object, object]:
            return self._inner.to_torch_tensors(trajectories)

    monkeypatch.setattr(trainer, "TorchTensorFactory", CpuOnlyFactory)

    stage_calls = {"count": 0}

    def fake_run_stage(self: object, p: object) -> list[Path]:
        del self, p
        stage_calls["count"] += 1
        return []

    monkeypatch.setattr(trainer.DiffusionTrainingPipeline, "_run_stage", fake_run_stage)

    trainer.DiffusionTrainingPipeline(
        dataset=str(dataset_path),
        output_path=str(tmp_path / "logs"),
        horizon=5,
        state_dim=3,
        n_diffusion_steps=16,
        epochs=1,
        batch_size=4,
        train_value=False,
    ).run()

    assert call_count["init"] == 1
    assert stage_calls["count"] == 1


def test_manage_stage_checkpoints_skips_ema_snapshot_when_nothing_is_saved(tmp_path: Path) -> None:
    from planning.diffusion.training import trainer

    class CountingEMA:
        def __init__(self) -> None:
            self.calls = 0

        def state_dict(self) -> dict[str, object]:
            self.calls += 1
            return {"weight": self.calls}

    class StubCheckpointManager:
        def __init__(self, root: Path) -> None:
            self._root = root

        def checkpoint_path(self, kind: str, epoch: int) -> Path:
            return self._root / kind / f"epoch_{epoch:04d}.ckpt"

    class RecordingManager(StubCheckpointManager):
        def __init__(self, root: Path) -> None:
            super().__init__(root)
            self.calls: list[dict[str, object]] = []

        def save(
            self,
            path: Path,
            *,
            model: object,
            normalizer: object,
            meta: dict[str, object],
            model_kind: str,
            ema_state_dict: dict[str, object] | None = None,
        ) -> Path:
            self.calls.append(
                {"path": path, "meta": meta, "kind": model_kind, "ema_state_dict": ema_state_dict}
            )
            return path

    ema = CountingEMA()
    manager = RecordingManager(tmp_path)
    params = trainer._StageParams(
        phase="diffusion",
        epoch_trainer=object(),
        model=object(),
        optimizer=object(),
        ema=ema,
        effective_epochs=10,
        patience=None,
        min_delta=0.0,
        log_every=1,
        config=object(),
        normalizer=PlannerStateNormalizer.identity(3),
        checkpoint_manager=manager,
        train_loader=object(),
        val_loader=None,
        summary_writer=None,
        checkpoint_every=0,
        latest_checkpoint_every=0,
        keep_last_checkpoints=0,
        extra_meta=None,
    )
    trainer._manage_stage_checkpoints(
        params,
        epoch=2,
        metric_loss=10.0,
        train_loss=10.0,
        is_new_best=False,
        recent_ckpts=deque(),
        ckpt_paths=[],
        latest_ckpt=tmp_path / "latest.ckpt",
        best_ckpt=tmp_path / "best.ckpt",
    )

    assert ema.calls == 0
    assert manager.calls == []


def test_manage_stage_checkpoints_reuses_single_ema_snapshot_when_writing(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from planning.diffusion.training import trainer

    class CountingEMA:
        def __init__(self) -> None:
            self.calls = 0

        def state_dict(self) -> dict[str, object]:
            self.calls += 1
            return {"weight": self.calls}

    class StubCheckpointManager:
        def __init__(self, root: Path) -> None:
            self._root = root

        def checkpoint_path(self, kind: str, epoch: int) -> Path:
            return self._root / kind / f"epoch_{epoch:04d}.ckpt"

    class RecordingManager(StubCheckpointManager):
        def __init__(self, root: Path) -> None:
            super().__init__(root)
            self.calls: list[dict[str, object]] = []

        def save(
            self,
            path: Path,
            *,
            model: object,
            normalizer: object,
            meta: dict[str, object],
            model_kind: str,
            ema_state_dict: dict[str, object] | None = None,
        ) -> Path:
            self.calls.append(
                {
                    "path": path,
                    "meta": meta,
                    "kind": model_kind,
                    "ema_state_dict": ema_state_dict,
                }
            )
            return path

    ema = CountingEMA()
    manager = RecordingManager(tmp_path)
    config = SimpleNamespace(
        dataset_key="toy",
        horizon=5,
        state_dim=3,
        n_diffusion_steps=8,
        n_hidden=16,
    )
    ckpt_paths: list[Path] = []

    params = trainer._StageParams(
        phase="diffusion",
        epoch_trainer=object(),
        model=object(),
        optimizer=object(),
        ema=ema,
        effective_epochs=2,
        patience=None,
        min_delta=0.0,
        log_every=1,
        config=config,
        normalizer=PlannerStateNormalizer.identity(3),
        checkpoint_manager=manager,
        train_loader=object(),
        val_loader=None,
        summary_writer=None,
        checkpoint_every=1,
        latest_checkpoint_every=1,
        keep_last_checkpoints=0,
        extra_meta=None,
    )
    trainer._manage_stage_checkpoints(
        params,
        epoch=1,
        metric_loss=1.0,
        train_loss=1.0,
        is_new_best=True,
        recent_ckpts=deque(),
        ckpt_paths=ckpt_paths,
        latest_ckpt=tmp_path / "latest.ckpt",
        best_ckpt=tmp_path / "best.ckpt",
    )

    assert len(manager.calls) == 3
    assert ema.calls == 1
    assert {id(call["ema_state_dict"]) for call in manager.calls} == {
        id(manager.calls[0]["ema_state_dict"])
    }


def test_diffusion_epoch_trainer_avoids_loss_item_calls_per_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")
    import torch

    from planning.diffusion.training.noise import DiffusionSchedule
    from planning.diffusion.training.trainer import DiffusionEpochTrainer

    class TinyDiffusionModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, noisy: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            del t
            return noisy * self.scale

    class LossProxy:
        def __init__(self, tensor: Any, counter: dict[str, int]) -> None:
            self._tensor = tensor
            self._counter = counter

        def backward(self) -> None:
            self._tensor.backward()

        def detach(self) -> Any:
            return self._tensor.detach()

        def item(self) -> float:
            self._counter["item_calls"] += 1
            return float(self._tensor.item())

    original_mse_loss = torch.nn.functional.mse_loss
    item_counter = {"item_calls": 0}

    def mse_loss_proxy(predicted: torch.Tensor, target: torch.Tensor) -> LossProxy:
        tensor = original_mse_loss(predicted, target)
        return LossProxy(tensor, item_counter)

    monkeypatch.setattr(torch.nn.functional, "mse_loss", mse_loss_proxy)

    observations = torch.randn(8, 5, 3)
    conditions = torch.zeros(8, 1)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(observations, conditions),
        batch_size=2,
        shuffle=False,
    )

    model = TinyDiffusionModel()
    trainer = DiffusionEpochTrainer(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=1e-3),
        schedule=DiffusionSchedule.linear(n_diffusion_steps=8),
    )

    train_loss = trainer.train_epoch(loader)
    val_loss = trainer.evaluate_epoch(loader)

    assert train_loss >= 0.0
    assert val_loss >= 0.0
    assert item_counter["item_calls"] == 0


def test_ema_accumulator_state_dict_includes_model_buffers() -> None:
    torch = pytest.importorskip("torch")
    from planning.diffusion.model import ValueModel
    from planning.diffusion.training.trainer import EMAAccumulator

    model = ValueModel(state_dim=3, horizon=8, dim=32)
    ema = EMAAccumulator(model)
    ema_state = ema.state_dict()
    model_state = model.state_dict()

    assert "net._const_t" in model_state
    assert "net._const_t" in ema_state
    assert torch.equal(ema_state["net._const_t"], model_state["net._const_t"])
