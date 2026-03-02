"""Script entrypoint for diffusion training workflow."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from planning.diffusion.config import PROJECT_CONFIG_DIR


@hydra.main(str(PROJECT_CONFIG_DIR), "diffusion_3d_training")
def main(cfg: DictConfig) -> None:
    from planning.diffusion.training import DiffusionTrainingPipeline

    values = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(values, dict):
        raise TypeError("Expected mapping config payload from Hydra.")
    ckpts = DiffusionTrainingPipeline(**values).run()
    print("Saved checkpoints:")
    for path in ckpts:
        print(f"- {path}")

if __name__ == "__main__":
    raise SystemExit(main())
