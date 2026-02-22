"""Regression tests for diffusion public API migration."""

from __future__ import annotations

import pytest


def test_legacy_public_api_names_removed() -> None:
    """Ensure legacy shim-like top-level entry points are not exported."""

    pytest.importorskip("planning.diffusion")
    import planning.diffusion as diffusion_pkg

    for legacy_name in ["train", "load_diffusion", "n_step_guided_p_sample"]:
        assert not hasattr(diffusion_pkg, legacy_name)
