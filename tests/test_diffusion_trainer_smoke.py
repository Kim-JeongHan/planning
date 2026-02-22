"""Smoke test for local diffuser training CLI path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def test_training_smoke(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from planning.diffusion.training.trainer import DiffusionTrainingPipeline

    dataset = np.random.RandomState(0).randn(24, 5, 3)
    dataset_path = tmp_path / "toy.npz"
    np.savez(dataset_path, observations=dataset)

    checkpoints = DiffusionTrainingPipeline(
        dataset=str(dataset_path),
        output_root=str(tmp_path / "logs"),
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


def test_training_package_exports_pipeline() -> None:
    from planning.diffusion.training import DiffusionTrainingPipeline as PackagePipeline
    from planning.diffusion.training.trainer import DiffusionTrainingPipeline as ModulePipeline

    assert PackagePipeline is ModulePipeline


def test_train_arg_resolver_parses_advanced_flags() -> None:
    from planning.diffusion.training.cli import TrainArgResolver

    values = TrainArgResolver(
        [
            "--dataset",
            "dummy.npz",
            "--diffusion-max-epochs",
            "2",
            "--value-patience",
            "3",
            "--diffusion-min-delta",
            "0.05",
            "--tensorboard-log-dir",
            "logs/tensorboard",
            "--checkpoint-every",
            "3",
            "--keep-last-checkpoints",
            "2",
            "--n-hidden",
            "64",
            "--device",
            "cuda:0",
        ]
    ).resolve()

    assert values["diffusion_max_epochs"] == 2
    assert values["value_patience"] == 3
    assert values["diffusion_min_delta"] == 0.05
    assert values["tensorboard_log_dir"] == "logs/tensorboard"
    assert values["checkpoint_every"] == 3
    assert values["keep_last_checkpoints"] == 2
    assert values["n_hidden"] == 64
    assert values["device"] == "cuda:0"


def test_resolve_training_device_auto_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    from planning.diffusion.training import trainer

    monkeypatch.setattr(trainer.torch.cuda, "is_available", lambda: False)
    resolved = trainer._resolve_training_device("auto")

    assert resolved.type == "cpu"


def test_resolve_training_device_rejects_cuda_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from planning.diffusion.training import trainer

    monkeypatch.setattr(trainer.torch.cuda, "is_available", lambda: False)

    with pytest.raises(ValueError, match="CUDA was requested"):
        trainer._resolve_training_device("cuda")


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
            self._inner = base_factory(normalizer)

        def to_torch_tensors(self, trajectories: np.ndarray) -> tuple[object, object]:
            return self._inner.to_torch_tensors(trajectories)

    monkeypatch.setattr(trainer, "TorchTensorFactory", CpuOnlyFactory)

    stage_calls = {"count": 0}

    def fake_run_diffusion_stage(self: object, **kwargs: object) -> list[Path]:
        del self, kwargs
        stage_calls["count"] += 1
        return []

    monkeypatch.setattr(
        trainer.DiffusionTrainingPipeline, "_run_diffusion_stage", fake_run_diffusion_stage
    )

    trainer.DiffusionTrainingPipeline(
        dataset=str(dataset_path),
        output_root=str(tmp_path / "logs"),
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

    class StubPathManager:
        def __init__(self, root: Path) -> None:
            self._root = root

        def checkpoint_path(self, kind: str, epoch: int) -> Path:
            return self._root / kind / f"epoch_{epoch:04d}.ckpt"

        def checkpoint_root(self, kind: str) -> Path:
            return self._root / kind

    class RecordingWriter:
        def __init__(self, path_manager: StubPathManager) -> None:
            self.path_manager = path_manager
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
    writer = RecordingWriter(StubPathManager(tmp_path))
    trainer._manage_stage_checkpoints(
        phase="diffusion",
        epoch=2,
        total_epochs=10,
        model=object(),
        ema=ema,
        normalizer=object(),
        checkpoint_writer=writer,
        config=object(),  # not used because no checkpoint write occurs
        dataset_key="toy",
        metric_loss=10.0,
        train_loss=10.0,
        is_new_best=False,
        recent_ckpts=[],
        ckpt_paths=[],
        latest_ckpt=tmp_path / "latest.ckpt",
        best_ckpt=tmp_path / "best.ckpt",
        checkpoint_every=0,
        latest_checkpoint_every=0,
        keep_last_checkpoints=0,
        extra_meta=None,
    )

    assert ema.calls == 0
    assert writer.calls == []


def test_manage_stage_checkpoints_reuses_single_ema_snapshot_when_writing(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from planning.diffusion.training import trainer

    class CountingEMA:
        def __init__(self) -> None:
            self.calls = 0

        def state_dict(self) -> dict[str, object]:
            self.calls += 1
            return {"weight": self.calls}

    class StubPathManager:
        def __init__(self, root: Path) -> None:
            self._root = root

        def checkpoint_path(self, kind: str, epoch: int) -> Path:
            return self._root / kind / f"epoch_{epoch:04d}.ckpt"

        def checkpoint_root(self, kind: str) -> Path:
            return self._root / kind

    class RecordingWriter:
        def __init__(self, path_manager: StubPathManager) -> None:
            self.path_manager = path_manager
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
    writer = RecordingWriter(StubPathManager(tmp_path))
    config = SimpleNamespace(horizon=5, state_dim=3, n_diffusion_steps=8, n_hidden=16)
    ckpt_paths: list[Path] = []

    trainer._manage_stage_checkpoints(
        phase="diffusion",
        epoch=1,
        total_epochs=2,
        model=object(),
        ema=ema,
        normalizer=object(),
        checkpoint_writer=writer,
        config=config,
        dataset_key="toy",
        metric_loss=1.0,
        train_loss=1.0,
        is_new_best=True,
        recent_ckpts=[],
        ckpt_paths=ckpt_paths,
        latest_ckpt=tmp_path / "latest.ckpt",
        best_ckpt=tmp_path / "best.ckpt",
        checkpoint_every=1,
        latest_checkpoint_every=1,
        keep_last_checkpoints=0,
        extra_meta=None,
    )

    assert len(writer.calls) == 3
    assert ema.calls == 1
    assert {id(call["ema_state_dict"]) for call in writer.calls} == {
        id(writer.calls[0]["ema_state_dict"])
    }


def test_diffusion_epoch_trainer_avoids_loss_item_calls_per_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    from planning.diffusion.training.noise import DiffusionSchedule
    from planning.diffusion.training.trainer import DiffusionEpochTrainer

    class TinyDiffusionModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, noisy: object, t: object) -> object:
            del t
            return noisy * self.scale

    class LossProxy:
        def __init__(self, tensor: object, counter: dict[str, int]) -> None:
            self._tensor = tensor
            self._counter = counter

        def backward(self) -> None:
            self._tensor.backward()

        def detach(self) -> object:
            return self._tensor.detach()

        def item(self) -> float:
            self._counter["item_calls"] += 1
            return float(self._tensor.item())

    original_mse_loss = torch.nn.functional.mse_loss
    item_counter = {"item_calls": 0}

    def mse_loss_proxy(predicted: object, target: object) -> LossProxy:
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
        torch_backend=torch,
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
