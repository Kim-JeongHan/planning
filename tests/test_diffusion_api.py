"""Regression tests for diffusion public API migration."""

from __future__ import annotations

import pytest


def test_torch_dependent_names_are_not_exported_from_package_root() -> None:
    """Keep the diffusion package root importable without the diffuser extra."""

    pytest.importorskip("planning.diffusion")
    import planning.diffusion as diffusion_pkg

    for legacy_name in [
        "CheckpointManager",
        "DiffusionArtifactLoader",
        "GuidedPolicy",
        "ValueGuide",
        "check_compatibility",
        "load_diffusion",
        "n_step_guided_p_sample",
        "train",
    ]:
        assert not hasattr(diffusion_pkg, legacy_name)
