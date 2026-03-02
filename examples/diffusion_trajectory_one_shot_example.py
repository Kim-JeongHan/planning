"""Generate a single diffusion rollout as a full trajectory."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import viser
import yaml
from pydantic import BaseModel, ValidationError, field_validator

from planning.collision import BoundedCollisionChecker, ObstacleCollisionChecker
from planning.constraint import select_collision_free_trajectory
from planning.diffusion import CheckpointManager, check_compatibility
from planning.diffusion.config import DiffusionInferenceConfig
from planning.diffusion.sampling import GuidedPolicy, ValueGuide
from planning.diffusion.utils import DiffusionArtifactLoader
from planning.map import Map
from planning.visualization import save_docs_image, setup_camera_top_view

DEFAULT_RUN_CONFIG_PATH = "config/diffusion_trajectory_one_shot.yaml"
DEFAULT_DOCS_IMAGE = "diffusion_trajectory_one_shot_example.png"



class EnvironmentConfig(BaseModel):
    """Map, obstacle, and start/goal configuration."""

    map_size: float = 24.0
    z_range: tuple[float, float] = (0.5, 2.5)
    obstacle_count: int = 12
    obstacle_min_size: float = 0.5
    obstacle_max_size: float = 2.0
    obstacle_color: tuple[int, int, int] = (200, 100, 50)
    start_state: tuple[float, float, float] = (8.0, 8.0, 2.0)
    goal_state: tuple[float, float, float] = (-8.0, -8.0, 1.0)

    @field_validator("obstacle_color", mode="before")
    @classmethod
    def validate_color(cls, value: object) -> tuple[int, int, int]:
        """Validate obstacle color as an RGB tuple."""
        if isinstance(value, str):
            raise ValueError("obstacle_color must be an iterable of 3 values")
        if not hasattr(value, "__len__"):
            raise ValueError("obstacle_color must be an iterable of 3 values")
        if len(value) != 3:  # type: ignore[arg-type]
            raise ValueError("obstacle_color must have exactly 3 values")
        return int(value[0]), int(value[1]), int(value[2])

    @property
    def start_state_array(self) -> np.ndarray:
        """Return start state as a float numpy vector."""
        return np.asarray(self.start_state, dtype=float)

    @property
    def goal_state_array(self) -> np.ndarray:
        """Return goal state as a float numpy vector."""
        return np.asarray(self.goal_state, dtype=float)


class RolloutConfig(BaseModel):
    """Validation parameters for sampled trajectories."""

    segment_resolution: float = 0.1
    endpoint_tolerance: float = 0.75


class DiffusionOneShotConfig(BaseModel):
    """Root config grouped by concerns."""

    diffusion: DiffusionInferenceConfig
    environment: EnvironmentConfig
    rollout: RolloutConfig = RolloutConfig()

    @classmethod
    def load(cls, config_path: str) -> DiffusionOneShotConfig:
        """Load and validate YAML configuration."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Run config not found: {path}")

        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"Run config must be a mapping: {path}")

        try:
            return cls.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(f"Invalid run config: {path}\n{exc}") from exc


def _visualize_selected_trajectory(
    server: viser.ViserServer,
    selected_trajectory: np.ndarray,
    start_state: np.ndarray,
    goal_state: np.ndarray,
) -> None:
    """Add selected trajectory and endpoints to viser scene."""
    server.scene.add_line_segments(
        "/diffusion/trajectory",
        points=np.stack([selected_trajectory[:-1], selected_trajectory[1:]], axis=1),
        colors=(50, 200, 100),
        line_width=2.5,
    )
    server.scene.add_icosphere(
        "/diffusion/start",
        radius=0.25,
        position=tuple(start_state[:3]),
        color=(0, 255, 0),
    )
    server.scene.add_icosphere(
        "/diffusion/goal",
        radius=0.25,
        position=tuple(goal_state[:3]),
        color=(255, 0, 0),
    )


def _register_save_image_hook(server: viser.ViserServer) -> None:
    """Register a callback to save a docs image on client connect."""

    @server.on_client_connect
    def handle_save(client: viser.ClientHandle) -> None:
        print("Saving image...")

        time.sleep(2)
        save_docs_image(client, DEFAULT_DOCS_IMAGE)
        print(f"Saved docs/images/{DEFAULT_DOCS_IMAGE}")


def main(
    run_config_path: str = DEFAULT_RUN_CONFIG_PATH,
    *,
    headless: bool = False,
    save_image: bool = False,
) -> np.ndarray | None:
    """Run one-shot trajectory generation and return selected trajectory."""
    config = DiffusionOneShotConfig.load(run_config_path)
    environment = config.environment
    start_state = environment.start_state_array
    goal_state = environment.goal_state_array
    print("=== Diffusion one-shot trajectory generation example ===")

    map_env = Map(size=environment.map_size, z_range=environment.z_range)
    bounds = np.asarray(map_env.get_bounds(), dtype=float)
    server = None
    if not headless:
        server = viser.ViserServer()
        print("Viser server started at http://localhost:8080")
        setup_camera_top_view(server)
        map_env.visualize_bounds(server)

    obstacles = map_env.generate_obstacles(
        server=server,
        num_obstacles=environment.obstacle_count,
        min_size=environment.obstacle_min_size,
        max_size=environment.obstacle_max_size,
        seed=config.diffusion.seed,
        color=environment.obstacle_color,
        check_overlap=True,
        obstacle_type="mixed",
    )
    collision_checker = BoundedCollisionChecker(
        ObstacleCollisionChecker(obstacles),
        bounds=bounds,
    )
    diffusion_config = config.diffusion
    diffusion_loader = DiffusionArtifactLoader(
        CheckpointManager.for_loading(
            diffusion_config.diffusion_checkpoint_path, device=diffusion_config.device
        ),
        seed=diffusion_config.seed,
    )
    value_loader = DiffusionArtifactLoader(
        CheckpointManager.for_loading(
            diffusion_config.value_checkpoint_path, device=diffusion_config.device
        ),
        seed=diffusion_config.seed,
    )
    diffusion_experiment = diffusion_loader.load(diffusion_config.diffusion_epoch)
    value_experiment = value_loader.load(diffusion_config.value_epoch)
    check_compatibility(diffusion_experiment, value_experiment)
    diffusion_experiment.ema.to(diffusion_config.device)
    value_experiment.ema.to(diffusion_config.device)
    guide = ValueGuide(model=value_experiment.ema, verbose=False)
    policy = GuidedPolicy(
        guide=guide,
        scale=diffusion_config.scale,
        diffusion_model=diffusion_experiment.ema,
        normalizer=diffusion_experiment.dataset.normalizer,
        preprocess_fns=[],
        n_guide_steps=diffusion_config.n_guide_steps,
        t_stopgrad=2,
        scale_grad_by_std=True,
        verbose=False,
    )
    # Fix start *and* goal via inpainting: {timestep_index: state_array}.
    # GuidedPolicy normalizes these to model space before denoising.
    conditions = {0: start_state, policy.horizon - 1: goal_state}

    print(f"Sampling trajectories with batch size {diffusion_config.sample_batch_size}...")
    sampling_start_time = time.perf_counter()
    result = policy(conditions, batch_size=config.diffusion.sample_batch_size, verbose=False)
    sampling_time = time.perf_counter() - sampling_start_time
    print(f"Sampling took {sampling_time:.2f} seconds.")
    print(f"Sampled {len(result.observations)} trajectories, checking constraints...")

    trajectory_candidates = np.asarray(result.observations, dtype=float)
    selected_trajectory = select_collision_free_trajectory(
        trajectories=trajectory_candidates,
        collision_checker=collision_checker,
        start_state=start_state,
        goal_state=goal_state,
        endpoint_tolerance=config.rollout.endpoint_tolerance,
        segment_resolution=config.rollout.segment_resolution,
    )

    if selected_trajectory is not None:
        print("Selected trajectory found.")
    else:
        start_dist = np.linalg.norm(
            trajectory_candidates[:, 0, :3] - start_state,
            axis=1,
        )
        goal_dist = np.linalg.norm(
            trajectory_candidates[:, -1, :3] - goal_state,
            axis=1,
        )
        endpoint_ok = int(
            ((start_dist <= config.rollout.endpoint_tolerance) &
             (goal_dist <= config.rollout.endpoint_tolerance)).sum()
        )
        print(
            f"No valid path. best start dist={start_dist.min():.3f}, "
            f"best goal dist={goal_dist.min():.3f}, endpoint candidates={endpoint_ok}/{len(trajectory_candidates)}"
        )
        print(
            f"No collision-free trajectory found "
            f"(batch_size={config.diffusion.sample_batch_size}, tolerance={config.rollout.endpoint_tolerance})."
        )
        return None

    print(f"Selected trajectory shape: {selected_trajectory.shape}")
    print(
        f"Selected trajectory first/last: {selected_trajectory[0]} -> {selected_trajectory[-1]}"
    )

    if server is not None:
        _visualize_selected_trajectory(
            server=server,
            selected_trajectory=selected_trajectory,
            start_state=start_state,
            goal_state=goal_state,
        )

        if save_image:
            _register_save_image_hook(server)

        print("Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Shutting down server.")

    return selected_trajectory


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate full trajectory from diffusion policy in one call."
    )
    parser.add_argument(
        "--run-config",
        type=str,
        default=DEFAULT_RUN_CONFIG_PATH,
        help="Run config path. Example: config/diffusion_trajectory_one_shot.yaml",
    )
    parser.add_argument("--headless", action="store_true", help="Run without viser visualizer.")
    parser.add_argument("--save-image", action="store_true", help="Save visualization image.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        run_config_path=args.run_config,
        headless=args.headless,
        save_image=args.save_image,
    )
