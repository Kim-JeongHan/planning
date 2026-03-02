"""Compatibility utilities for the local ``diffuser`` implementation."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path

import torch  # type: ignore

from .core import DiffusionDataset, DiffusionExperiment, PlannerStateNormalizer
from .training.checkpoint import CheckpointManager


def _load_model_from_payload(payload: dict[str, object], expected: str) -> object:
    return ModelLoader(expected).load(payload)


@dataclass(frozen=True)
class _ResolvedModelDescriptor:
    module_path: str
    class_name: str


class ModelLoader:
    """Load model class and state from persisted payload."""

    def __init__(self, expected: str) -> None:
        self.expected = expected

    @staticmethod
    def _extract_descriptor(payload: dict[str, object]) -> _ResolvedModelDescriptor:
        model_class_path = payload.get("model_class_path", "")
        if not model_class_path:
            raise ValueError("Checkpoint missing 'model_class_path'")
        module_path, _, class_name = model_class_path.rpartition(".")
        if not module_path:
            raise ValueError("Invalid model_class_path in checkpoint payload.")
        return _ResolvedModelDescriptor(module_path=module_path, class_name=class_name)

    def _resolve_module(self, module_path: str) -> object:
        return importlib.import_module(module_path)

    def _load_model_cls(
        self, payload: dict[str, object], descriptor: _ResolvedModelDescriptor
    ) -> object:
        module = self._resolve_module(descriptor.module_path)
        model_kwargs = payload.get("model_kwargs", {})
        model_cls = getattr(module, descriptor.class_name)
        return model_cls(**model_kwargs)  # type: ignore[misc]

    @staticmethod
    def _load_state_dict(payload: dict[str, object]) -> object:
        state = payload.get("ema_state_dict") or payload.get("model_state_dict")
        if state is None:
            state = payload.get("state_dict")
        if not state:
            raise ValueError("Checkpoint missing model state dict")
        return state

    def load(self, payload: dict[str, object]) -> object:
        descriptor = self._extract_descriptor(payload)
        model = self._load_model_cls(payload, descriptor)
        state = self._load_state_dict(payload)
        missing, unexpected = model.load_state_dict(state, strict=False)  # type: ignore[arg-type]
        if missing:
            raise ValueError(
                f"Checkpoint has missing keys for {self.expected} model: {missing}"
            )
        if unexpected:
            # Incompatible but still usable for strict local workflows.
            pass
        model.eval()
        return model


class DiffusionArtifactLoader:
    """Load checkpoints into DiffusionExperiment objects."""

    def __init__(self, checkpoint_manager: CheckpointManager, *, seed: int | None = None) -> None:
        self.checkpoint_manager = checkpoint_manager
        self.seed = seed

    def resolve(self, epoch: str | int) -> Path:
        return self.checkpoint_manager.resolve(epoch)

    def load(self, epoch: str | int = "latest") -> DiffusionExperiment:
        checkpoint_file, payload = self.checkpoint_manager.load(epoch)

        if self.seed is not None:
            torch.manual_seed(self.seed)

        meta = dict(payload.get("meta", {}))
        meta["path"] = str(checkpoint_file)
        normalizer_payload = payload.get("normalizer")
        if normalizer_payload is None:
            state_dim = int(meta.get("state_dim", 3))
            normalizer = PlannerStateNormalizer.identity(state_dim)
        else:
            normalizer = PlannerStateNormalizer.from_dict(normalizer_payload)

        horizon = int(meta.get("horizon", 64))
        state_dim = int(meta.get("state_dim", normalizer.mean.shape[0]))
        dataset_name = str(meta.get("dataset", "dataset"))
        model = _load_model_from_payload(payload, expected="DiffusionExperiment")
        model.horizon = int(getattr(model, "horizon", horizon))
        model.state_dim = int(getattr(model, "state_dim", state_dim))
        if self.seed is not None and hasattr(model, "set_seed"):
            model.set_seed(self.seed)
        if hasattr(model, "meta"):
            merged_meta = dict(model.meta)
            merged_meta.update(meta)
            model.meta = merged_meta
        else:
            model.meta = meta

        payload_dataset = DiffusionDataset(
            name=dataset_name,
            normalizer=normalizer,
            horizon=horizon,
            state_dim=state_dim,
        )
        return DiffusionExperiment(ema=model, dataset=payload_dataset, meta=meta)


def check_compatibility(
    diffusion_experiment: DiffusionExperiment, value_experiment: DiffusionExperiment
) -> None:
    """Validate state dimension and horizon compatibility."""
    if diffusion_experiment.dataset.state_dim != value_experiment.dataset.state_dim:
        raise ValueError(
            "Diffusion/value state dimension mismatch: "
            f"{diffusion_experiment.dataset.state_dim} vs {value_experiment.dataset.state_dim}"
        )
    if diffusion_experiment.dataset.horizon != value_experiment.dataset.horizon:
        raise ValueError(
            "Diffusion/value horizon mismatch: "
            f"{diffusion_experiment.dataset.horizon} vs {value_experiment.dataset.horizon}"
        )

    if (
        diffusion_experiment.dataset.normalizer.mean.shape
        != value_experiment.dataset.normalizer.mean.shape
    ):
        raise ValueError("Diffusion/value normalizer shape mismatch.")


class Config:
    """Dynamic class factory with string class path loading."""

    def __init__(self, class_path: str, **kwargs: object) -> None:
        self.class_path = class_path
        self.kwargs = kwargs

    def _resolve_module(self, class_path: str) -> tuple[str, str]:
        module_path, _, class_name = class_path.rpartition(".")
        if not module_path:
            raise ValueError(f"Invalid class path: {class_path!r}")

        if (
            "." not in module_path
            and (
                module_path.startswith("sampling")
                or module_path.startswith("utils")
                or module_path.startswith("training")
            )
        ):
            module_path = f"planning.diffusion.{module_path}"

        return module_path, class_name

    def __call__(self) -> object:
        module_path, class_name = self._resolve_module(self.class_path)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name, None)
        if cls is None:
            raise AttributeError(f"{class_name!r} not found in {module_path!r}")
        return cls(**self.kwargs)
