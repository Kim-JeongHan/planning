"""Tests for diffuser utility helpers."""

from pathlib import Path

import pytest

from planning.diffusion.utils import _load_templating_context, _resolve_template_path


def test_load_templating_context_from_yaml_file(tmp_path: Path) -> None:
    """YAML config should populate horizon, diffusion steps, and discount."""
    pytest.importorskip("yaml")

    cfg_path = tmp_path / "diffusion.yaml"
    cfg_path.write_text("horizon: 12\nn_diffusion_steps: 32\ndiscount: 0.75\n", encoding="utf-8")

    context = _load_templating_context(
        str(cfg_path),
        "unit-dataset",
        fallback={"horizon": 64, "n_diffusion_steps": 100, "discount": 0.99},
    )

    assert context["dataset"] == "unit-dataset"
    assert context["horizon"] == 12
    assert context["n_diffusion_steps"] == 32
    assert context["discount"] == 0.75


def test_template_resolution_uses_yaml(tmp_path: Path) -> None:
    """YAML config should resolve diffusion-style template paths."""
    pytest.importorskip("yaml")

    cfg_path = tmp_path / "diffusion.yaml"
    cfg_path.write_text("horizon: 8\nn_diffusion_steps: 16\ndiscount: 0.9\n", encoding="utf-8")

    path = _resolve_template_path(
        "f:diffusion/defaults_H{horizon}_T{n_diffusion_steps}_d{discount}",
        dataset="unit",
        config=str(cfg_path),
    )

    assert path == "diffusion/defaults_H8_T16_d0.9"


def test_template_resolution_infers_from_disk(tmp_path: Path) -> None:
    """Without config, fallback to matching checkpoint directory under loadbase."""
    dataset_dir = tmp_path / "unit-dataset" / "diffusion" / "defaults_H16_T16_d1.0"
    dataset_dir.mkdir(parents=True)

    path = _resolve_template_path(
        "f:diffusion/defaults_H{horizon}_T{n_diffusion_steps}_d{discount}",
        dataset="unit-dataset",
        loadbase=str(tmp_path),
        config=None,
    )

    assert path == "diffusion/defaults_H16_T16_d1.0"
