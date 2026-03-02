"""Compatibility utilities for the local ``diffuser`` implementation."""

from __future__ import annotations

import importlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch  # type: ignore
import yaml  # type: ignore

from .core import DiffusionDataset, DiffusionExperiment, PlannerStateNormalizer

_CHECKPOINT_FILE_RE = re.compile(r"epoch_(\d+)\.ckpt")
_CKPT_FILE_RE = re.compile(r"ckpt_(\d+)\.pt")
_LATEST_CKPT_FILE_RE = re.compile(r"latest\.ckpt")
_BEST_CKPT_FILE_RE = re.compile(r"best\.ckpt")
_TEMPLATE_TOKEN_RE = re.compile(r"\{(\w+)\}")
_TEMPLATE_CONTEXT_FALLBACK: dict[str, object] = {
    "horizon": 64,
    "n_diffusion_steps": 100,
    "discount": 0.99,
    "H": 64,
    "T": 100,
}
_LOGGER = logging.getLogger(__name__)


def _extract_checkpoint_epoch(path: Path) -> int | None:
    name = path.name
    if _LATEST_CKPT_FILE_RE.fullmatch(name) is not None:
        return int(1e9)
    if _BEST_CKPT_FILE_RE.fullmatch(name) is not None:
        return int(1e9) - 1
    for pattern in (_CHECKPOINT_FILE_RE, _CKPT_FILE_RE):
        match = pattern.fullmatch(name)
        if match is not None:
            return int(match.group(1))
    return None


def _collect_checkpoint_files(root: Path) -> list[tuple[int, Path]]:
    epochs: list[tuple[int, Path]] = []
    for path in root.glob("*.ckpt"):
        epoch = _extract_checkpoint_epoch(path)
        if epoch is not None:
            epochs.append((epoch, path))
    for path in root.glob("*.pt"):
        epoch = _extract_checkpoint_epoch(path)
        if epoch is not None:
            epochs.append((epoch, path))
    return sorted(epochs, key=lambda item: item[0])


def _select_best_checkpoint_by_loss(candidates: list[tuple[int, Path]]) -> Path | None:
    """Select checkpoint with minimum `meta.loss`; return None if unavailable."""
    best_loss = float("inf")
    best_path: Path | None = None
    for _, candidate_path in candidates:
        try:
            payload = _load_checkpoint(candidate_path)
        except Exception as exc:
            _LOGGER.debug("Skipping invalid checkpoint %s: %s", candidate_path, exc)
            continue
        meta = payload.get("meta", {})
        raw_loss = meta.get("loss")
        try:
            loss_value = float(raw_loss)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(loss_value):
            continue
        if loss_value < best_loss:
            best_loss = loss_value
            best_path = candidate_path
    if best_path is not None:
        return best_path
    if candidates:
        return candidates[-1][1]
    return None


def _load_yaml_templating_context(
    config_path: str, fallback: dict[str, object]
) -> dict[str, object]:
    path = Path(config_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not path.is_file():
        raise FileNotFoundError(f"Config path is not a file: {config_path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError(
            "Diffusion YAML config must be a mapping (for example: key-value pairs)."
        )

    context = {
        name: payload.get(name, fallback[name])
        for name in ("horizon", "n_diffusion_steps", "discount")
        if name in payload or name in fallback
    }
    if "horizon" not in context:
        context["horizon"] = fallback["horizon"]
    if "n_diffusion_steps" not in context:
        context["n_diffusion_steps"] = fallback["n_diffusion_steps"]
    if "discount" not in context:
        context["discount"] = fallback["discount"]
    return context


def _load_python_templating_context(
    python_path: Path, fallback: dict[str, object]
) -> dict[str, object]:
    """Load templating variables from a Python module file."""
    spec = importlib.util.spec_from_file_location(
        f"_diffuser_template_{python_path.stem}_{id(python_path)}", python_path
    )
    if spec is None or spec.loader is None:
        return {
            "horizon": fallback["horizon"],
            "n_diffusion_steps": fallback["n_diffusion_steps"],
            "discount": fallback["discount"],
        }

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    context = {
        name: getattr(module, name)
        for name in ("horizon", "n_diffusion_steps", "discount")
        if hasattr(module, name)
    }
    if "horizon" not in context:
        context["horizon"] = fallback["horizon"]
    if "n_diffusion_steps" not in context:
        context["n_diffusion_steps"] = fallback["n_diffusion_steps"]
    if "discount" not in context:
        context["discount"] = fallback["discount"]
    return context


def _resolve_python_config_fallback(module: str) -> Path | None:
    root = Path(__file__).resolve().parents[1]
    candidate = root / Path(*module.split("."))
    if candidate.with_suffix(".py").exists():
        return candidate.with_suffix(".py")
    if candidate.exists() and candidate.is_file():
        return candidate
    return None


def _normalize_legacy_module_path(module_path: str) -> str:
    """Normalize legacy local package module paths to the new planning namespace."""
    legacy_prefix = "diffuser."
    new_prefix = "planning.diffusion."
    training_model_prefix = "planning.diffusion.training.model"
    model_prefix = "planning.diffusion.model"
    if module_path.startswith(legacy_prefix):
        return f"{new_prefix}{module_path[len(legacy_prefix):]}"
    if module_path == training_model_prefix:
        return model_prefix
    return module_path


def _load_templating_context(
    config: str | None, dataset: str, *, fallback: dict[str, object]
) -> dict[str, object]:
    if config is None:
        return {"dataset": dataset, **fallback}

    path = Path(config)
    if path.suffix.lower() in {".yml", ".yaml"}:
        context = _load_yaml_templating_context(config, fallback)
        context["dataset"] = dataset
        context = {"dataset": dataset, **fallback, **context}
        return context
    if path.suffix.lower() == ".py":
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config}")
        context = _load_python_templating_context(path.expanduser(), fallback)
        context["dataset"] = dataset
        return {"dataset": dataset, **fallback, **context}

    # Keep this import lazy to avoid hard dependency at import time.
    try:
        spec = importlib.util.find_spec(config)
    except ModuleNotFoundError:
        spec = None
    if spec is None:
        fallback_path = _resolve_python_config_fallback(config)
        if fallback_path is not None and fallback_path.exists():
            context = _load_python_templating_context(fallback_path.expanduser(), fallback)
            context["dataset"] = dataset
            return {"dataset": dataset, **fallback, **context}
        return {"dataset": dataset, **fallback}

    module = importlib.import_module(config)
    attrs = [
        name
        for name in ("horizon", "n_diffusion_steps", "discount")
        if hasattr(module, name)
    ]
    context = {"dataset": dataset, **fallback}
    for name in attrs:
        context[name] = getattr(module, name)
    return context


def _template_to_glob_pattern(template_body: str) -> str:
    return _TEMPLATE_TOKEN_RE.sub("*", template_body)


def _template_to_regex(template_body: str) -> re.Pattern[str]:
    parts = _TEMPLATE_TOKEN_RE.split(template_body)
    regex_parts: list[str] = []
    for index, part in enumerate(parts):
        if index % 2 == 0:
            regex_parts.append(re.escape(part))
        else:
            regex_parts.append(rf"(?P<{part}>[^/]+)")
    return re.compile(r"^" + "".join(regex_parts) + r"$")


def _infer_template_context_from_disk(
    *,
    loadbase: str,
    dataset: str,
    template_body: str,
    fallback: dict[str, object],
) -> dict[str, object] | None:
    """Infer template values from existing checkpoint directories under loadbase/dataset."""
    base = Path(loadbase) / dataset
    if not base.exists() or not base.is_dir():
        return None

    pattern = _template_to_glob_pattern(template_body)
    pattern_re = _template_to_regex(template_body)
    candidates = sorted(base.glob(pattern))
    if not candidates:
        return None

    for candidate in candidates:
        relative = candidate.relative_to(base).as_posix()
        match = pattern_re.fullmatch(relative)
        if match is None:
            continue

        context: dict[str, object] = {"dataset": dataset, **fallback}
        try:
            values = match.groupdict()
            if "horizon" in values and values["horizon"] is not None:
                context["horizon"] = int(values["horizon"])
                context["H"] = int(values["horizon"])
            if "H" in values and values["H"] is not None:
                context["horizon"] = int(values["H"])
                context["H"] = int(values["H"])
            if "n_diffusion_steps" in values and values["n_diffusion_steps"] is not None:
                context["n_diffusion_steps"] = int(values["n_diffusion_steps"])
                context["T"] = int(values["n_diffusion_steps"])
            if "T" in values and values["T"] is not None:
                context["n_diffusion_steps"] = int(values["T"])
                context["T"] = int(values["T"])
            if "discount" in values and values["discount"] is not None:
                context["discount"] = float(values["discount"])
        except ValueError:
            continue

        return context

    return None


def _resolve_template_path(
    template: str,
    *,
    dataset: str,
    config: str | None = None,
    loadbase: str | None = None,
    fallback: dict[str, object] | None = None,
) -> str:
    """Resolve ``f:...`` style templates into concrete relative paths."""
    if fallback is None:
        fallback = _TEMPLATE_CONTEXT_FALLBACK

    if not template.startswith("f:"):
        return template

    context: dict[str, object] = {"dataset": dataset, **fallback}
    template_body = template[2:]
    if config is not None:
        context = _load_templating_context(
            config,
            dataset,
            fallback=fallback,
        )
    elif loadbase is not None:
        context = _infer_template_context_from_disk(
            loadbase=loadbase,
            dataset=dataset,
            template_body=template_body,
            fallback=fallback,
        ) or context

    context["H"] = context["horizon"]
    context["T"] = context["n_diffusion_steps"]
    try:
        return template_body.format(**context)
    except KeyError as exc:
        raise ValueError(
            "Cannot resolve diffusion template path; check config variables."
            f" Missing key {exc}"
        ) from exc


def _candidate_checkpoint_root(loadbase: str, dataset: str, loadpath: str) -> list[Path]:
    base = Path(loadbase)
    candidates = [base / dataset / loadpath, Path(loadpath)]
    return [path for path in candidates if path.exists()]


def _resolve_named_checkpoint(
    root: Path, requested: str, candidates: list[tuple[int, Path]]
) -> Path | None:
    if requested == "latest":
        direct_latest = root / "latest.ckpt"
        if direct_latest.exists():
            return direct_latest
        return None
    if requested == "best":
        direct_best = root / "best.ckpt"
        if direct_best.exists():
            return direct_best
        return _select_best_checkpoint_by_loss(candidates)
    return None


def _resolve_checkpoint_by_epoch(
    root: Path, target_epoch: int, candidates: list[tuple[int, Path]]
) -> Path | None:
    direct_ckpt = root / f"epoch_{target_epoch:04d}.ckpt"
    if direct_ckpt.exists():
        return direct_ckpt
    direct_alt = root / f"ckpt_{target_epoch:04d}.pt"
    if direct_alt.exists():
        return direct_alt
    for candidate_epoch, candidate_path in candidates:
        if candidate_epoch == target_epoch:
            return candidate_path
    return None


def _resolve_checkpoint_fallback(
    root: Path, candidates: list[tuple[int, Path]]
) -> Path | None:
    candidates_only = [path for _, path in candidates]
    if candidates_only:
        return candidates_only[-1]

    checkpoint_json = root / "checkpoint.json"
    if not checkpoint_json.exists():
        return None
    payload = json.loads(checkpoint_json.read_text(encoding="utf-8"))
    latest = payload.get("latest_checkpoint")
    if not latest:
        return None
    explicit = root / latest
    if explicit.exists():
        return explicit
    return None


def _coerce_epoch_value(value: str | int) -> int | None:
    if isinstance(value, int):
        return int(value)
    normalized = value.strip().lower()
    try:
        return int(normalized)
    except ValueError:
        return None


def _resolve_checkpoint_file(  # noqa: C901
    loadbase: str,
    dataset: str,
    loadpath: str,
    epoch: str | int,
    config: str | None,
) -> Path:
    resolved = _resolve_template_path(
        loadpath, dataset=dataset, config=config, loadbase=loadbase
    )
    roots = _candidate_checkpoint_root(loadbase, dataset, resolved)
    if not roots:
        raise FileNotFoundError(
            "Could not resolve checkpoint root for "
            f"loadbase={loadbase!r}, dataset={dataset!r}, loadpath={loadpath!r}"
        )

    root = roots[0]
    if root.is_file():
        return root

    if not root.exists():
        raise FileNotFoundError(f"Checkpoint root does not exist: {root}")

    candidates = _collect_checkpoint_files(root)
    requested = ""
    if isinstance(epoch, str):
        requested = epoch.strip().lower()
        if requested:
            by_name = _resolve_named_checkpoint(root, requested, candidates)
            if by_name is not None:
                return by_name
            direct = root / requested
            if direct.exists():
                return direct
    target_epoch = _coerce_epoch_value(epoch)
    if target_epoch is not None:
        by_epoch = _resolve_checkpoint_by_epoch(root, target_epoch, candidates)
        if by_epoch is not None:
            return by_epoch

    fallback = _resolve_checkpoint_fallback(root, candidates)
    if fallback is not None:
        return fallback

    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {root}")
    if not requested:
        return candidates[-1][1]
    raise FileNotFoundError(
        f"Could not find epoch {epoch} in {root}; "
        f"available: {[path.name for _, path in candidates]}"
    )


def _load_checkpoint(file_path: Path) -> dict[str, object]:
    if file_path.suffix.lower() in {".pt", ".ckpt"}:
        try:
            return torch.load(file_path, map_location="cpu")
        except Exception as exc:
            raise RuntimeError(f"Failed to load torch checkpoint: {file_path}") from exc
    if file_path.suffix.lower() == ".npz":
        payload = dict(np.load(file_path, allow_pickle=True))
        normalized: dict[str, object] = {}
        for key, value in payload.items():
            if isinstance(value, np.ndarray) and value.dtype == object:
                normalized[key] = value.item()
            else:
                normalized[key] = value
        return normalized
    raise ValueError(f"Unsupported checkpoint format: {file_path.suffix}")


def _load_model_from_payload(payload: dict[str, object], expected: str) -> object:
    return ModelPayloadLoader(expected).load(payload)


@dataclass(frozen=True)
class _ResolvedModelDescriptor:
    module_path: str
    class_name: str


class ModelPayloadLoader:
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
        try:
            return importlib.import_module(module_path)
        except ModuleNotFoundError:
            return importlib.import_module(_normalize_legacy_module_path(module_path))

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


class TemplatingContextResolver:
    """Resolve checkpoint template variables with fallback semantics."""

    def __init__(self, fallback: dict[str, object] | None = None) -> None:
        self.fallback = dict(_TEMPLATE_CONTEXT_FALLBACK)
        if fallback:
            self.fallback.update(fallback)

    def resolve(
        self,
        template: str,
        *,
        dataset: str,
        config: str | None = None,
        loadbase: str | None = None,
    ) -> str:
        return _resolve_template_path(
            template,
            dataset=dataset,
            config=config,
            loadbase=loadbase,
            fallback=self.fallback,
        )



class CheckpointCatalog:
    """Resolve checkpoint root and enumerate candidates."""

    def __init__(
        self,
        loadbase: str,
        dataset: str,
        loadpath: str,
        config: str | None = None,
    ) -> None:
        self.loadbase = loadbase
        self.dataset = dataset
        self.loadpath = loadpath
        self.config = config
        self.template_resolver = TemplatingContextResolver(_TEMPLATE_CONTEXT_FALLBACK)

    @property
    def root(self) -> Path:
        resolved = self.template_resolver.resolve(
            self.loadpath,
            dataset=self.dataset,
            config=self.config,
            loadbase=self.loadbase,
        )
        roots = _candidate_checkpoint_root(self.loadbase, self.dataset, resolved)
        if not roots:
            raise FileNotFoundError(
                f"Could not resolve checkpoint root for loadbase={self.loadbase!r}, "
                f"dataset={self.dataset!r}, loadpath={self.loadpath!r}"
            )
        return roots[0]

    def candidates(self) -> list[tuple[int, Path]]:
        if self.root.is_file():
            return []
        return _collect_checkpoint_files(self.root)


class DiffusionArtifactLoader:
    """Load checkpoints into DiffusionExperiment objects."""

    def __init__(self, catalog: CheckpointCatalog, *, seed: int | None = None) -> None:
        self.catalog = catalog
        self.seed = seed

    def resolve(self, epoch: str | int) -> Path:
        return _resolve_checkpoint_file(
            self.catalog.loadbase,
            self.catalog.dataset,
            self.catalog.loadpath,
            epoch,
            self.catalog.config,
        )

    def load(self, epoch: str | int = "latest") -> DiffusionExperiment:
        checkpoint_file = self.resolve(epoch)
        payload = _load_checkpoint(checkpoint_file)

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
        dataset_name = str(meta.get("dataset", self.catalog.dataset))
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
    """Dynamic class factory with string class path compatibility."""

    def __init__(self, class_path: str, **kwargs: object) -> None:
        self.class_path = class_path
        self.kwargs = kwargs

    def _resolve_module(self, class_path: str) -> tuple[str, str]:
        module_path, _, class_name = class_path.rpartition(".")
        if not module_path:
            raise ValueError(f"Invalid class path: {class_path!r}")
        module_path = _normalize_legacy_module_path(module_path)

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
