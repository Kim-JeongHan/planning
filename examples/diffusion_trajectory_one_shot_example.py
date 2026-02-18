"""Generate a single diffusion rollout as a full trajectory."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import viser

from planning.collision import CollisionChecker, ObstacleCollisionChecker
from planning.map import Map
from planning.visualization import save_docs_image, setup_camera_top_view


def _load_run_config(config_path: str) -> dict[str, object]:
    """Load YAML configuration for the example."""
    try:
        import yaml
    except Exception as exc:
        raise ImportError(
            "PyYAML is required to load diffuser trajectory example config."
        ) from exc

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Run config not found: {path}")

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Run config must be a mapping: {path}")
    return payload


def _ensure_vector(values: object, dim: int, name: str) -> np.ndarray:
    """Convert YAML value into a fixed-size vector."""
    array = np.asarray(values, dtype=float)
    if array.shape != (dim,):
        raise ValueError(f"{name} must have shape ({dim},), got {array.shape}")
    return array


def _ensure_vector2(values: object, name: str) -> tuple[float, float]:
    """Convert YAML value into a 2D tuple."""
    array = _ensure_vector(values, 2, name)
    return float(array[0]), float(array[1])


def _extract_observations(result: object) -> np.ndarray:
    """Extract trajectory observations from policy output."""
    if hasattr(result, "observations"):
        return np.asarray(result.observations, dtype=float)
    if isinstance(result, tuple):
        for item in result:
            observations = _extract_observations(item)
            if observations.ndim == 3:
                return observations
        raise ValueError("Policy result tuple did not contain trajectory observations.")
    return np.asarray(result, dtype=float)


def _is_within_bounds(state: np.ndarray, bounds: np.ndarray) -> bool:
    """Return True when a state is inside map bounds."""
    return bool(np.all(state[:3] >= bounds[:, 0]) and np.all(state[:3] <= bounds[:, 1]))


def _trajectory_is_collision_free(
    trajectory: np.ndarray,
    collision_checker: CollisionChecker,
    bounds: np.ndarray,
    segment_resolution: float,
    start_state: np.ndarray | None = None,
    goal_state: np.ndarray | None = None,
    endpoint_tolerance: float = 0.5,
) -> bool:
    """Validate a trajectory against bounds, endpoints, and collisions."""
    if trajectory.ndim != 2 or trajectory.shape[1] < 3 or len(trajectory) < 2:
        return False

    points = trajectory[:, :3]
    if not all(_is_within_bounds(point, bounds) for point in points):
        return False

    start_gap = (
        np.linalg.norm(points[0] - start_state[:3], ord=2)
        if start_state is not None
        else 0.0
    )
    if start_gap > endpoint_tolerance:
        return False

    goal_gap = (
        np.linalg.norm(points[-1] - goal_state[:3], ord=2)
        if goal_state is not None
        else 0.0
    )
    if goal_gap > endpoint_tolerance:
        return False

    for idx in range(points.shape[0] - 1):
        if not collision_checker.is_path_collision_free(
            points[idx],
            points[idx + 1],
            resolution=segment_resolution,
        ):
            return False
    return True


def _select_collision_free_trajectory(
    trajectories: np.ndarray,
    collision_checker: CollisionChecker,
    bounds: np.ndarray,
    *,
    start_state: np.ndarray | None = None,
    goal_state: np.ndarray | None = None,
    endpoint_tolerance: float = 0.5,
    segment_resolution: float = 0.1,
) -> np.ndarray | None:
    """Select first valid trajectory from a batch."""
    for trajectory in trajectories:
        if _trajectory_is_collision_free(
            trajectory,
            collision_checker,
            bounds,
            segment_resolution=segment_resolution,
            start_state=start_state,
            goal_state=goal_state,
            endpoint_tolerance=endpoint_tolerance,
        ):
            return np.asarray(trajectory[:, :3], dtype=float)
    return None


def _build_diffusion_policy(
    *,
    loadbase: str,
    dataset: str,
    config: str | None = None,
    diffusion_loadpath: str,
    value_loadpath: str,
    diffusion_epoch: str | int,
    value_epoch: str | int,
    n_guide_steps: int,
    scale: float,
    seed: int | None,
) -> object:
    """Build diffusion policy from local checkpoints."""
    try:
        from planning.diffusion import check_compatibility
        from planning.diffusion.utils import CheckpointCatalog, DiffusionArtifactLoader
        from planning.diffusion.sampling import GuidedPolicy, ValueGuide
    except Exception as exc:
        raise RuntimeError(
            "Optional diffuser dependencies are required. "
            "Install with `uv sync --extra diffuser`."
        ) from exc

    diffusion_loader = DiffusionArtifactLoader(
        CheckpointCatalog(loadbase, dataset=dataset, loadpath=diffusion_loadpath, config=config),
        seed=seed,
    )
    value_loader = DiffusionArtifactLoader(
        CheckpointCatalog(loadbase, dataset=dataset, loadpath=value_loadpath, config=config),
        seed=seed,
    )
    diffusion_experiment = diffusion_loader.load(diffusion_epoch)
    value_experiment = value_loader.load(value_epoch)
    check_compatibility(diffusion_experiment, value_experiment)

    guide = ValueGuide(model=value_experiment.ema, verbose=False)
    return GuidedPolicy(
        guide=guide,
        scale=scale,
        diffusion_model=diffusion_experiment.ema,
        normalizer=diffusion_experiment.dataset.normalizer,
        preprocess_fns=[],
        n_guide_steps=n_guide_steps,
        t_stopgrad=2,
        scale_grad_by_std=True,
        verbose=False,
    )


def _sample_trajectories(
    policy: object,
    condition: np.ndarray,
    sample_batch_size: int,
    condition_key: int | str = 0,
) -> np.ndarray:
    """Run diffusion policy once and return a trajectory batch."""
    result = policy({condition_key: condition}, batch_size=sample_batch_size, verbose=False)
    observations = _extract_observations(result)
    if observations.ndim != 3:
        raise ValueError("Policy output must be [batch, horizon, state_dim].")
    return observations


def main(
    run_config_path: str = "config/diffusion_trajectory_one_shot.yaml",
    headless: bool = False,
    save_image: bool = False,
) -> np.ndarray | None:
    """Run one-shot trajectory generation and return selected trajectory."""
    cfg = _load_run_config(run_config_path)
    print("=== Diffusion one-shot trajectory generation example ===")

    seed = int(cfg.get("seed", 42))
    loadbase = str(cfg.get("loadbase", "logs/pretrained"))
    dataset = str(cfg["dataset"])
    config = cfg.get("config")
    config = str(config) if config is not None else None
    diffusion_loadpath = str(
        cfg.get("diffusion_loadpath", "f:diffusion/defaults_H{horizon}_T{n_diffusion_steps}")
    )
    value_loadpath = str(
        cfg.get("value_loadpath", "f:values/defaults_H{horizon}_T{n_diffusion_steps}_d{discount}")
    )
    diffusion_epoch = cfg.get("diffusion_epoch", "latest")
    value_epoch = cfg.get("value_epoch", "latest")
    n_guide_steps = int(cfg.get("n_guide_steps", 2))
    scale = float(cfg.get("scale", 0.1))
    sample_batch_size = int(cfg.get("sample_batch_size", 4))
    map_size = float(cfg.get("map_size", 24))
    z_range = _ensure_vector2(cfg.get("z_range", (0.5, 2.5)), "z_range")
    obstacle_count = int(cfg.get("obstacle_count", 12))
    obstacle_min_size = float(cfg.get("obstacle_min_size", 0.5))
    obstacle_max_size = float(cfg.get("obstacle_max_size", 2.0))
    obstacle_color = tuple(int(v) for v in cfg.get("obstacle_color", (200, 100, 50)))
    start_state = _ensure_vector(cfg.get("start_state", (8.0, 8.0, 2.0)), 3, "start_state")
    goal_state = _ensure_vector(cfg.get("goal_state", (-8.0, -8.0, 1.0)), 3, "goal_state")
    condition_key = cfg.get("condition_key", 0)
    if not isinstance(condition_key, (int, str)):
        raise TypeError("condition_key must be int or str")
    segment_resolution = float(cfg.get("segment_resolution", 0.1))
    endpoint_tolerance = float(cfg.get("endpoint_tolerance", 0.75))
    if not isinstance(diffusion_epoch, (int, str)):
        raise TypeError("diffusion_epoch must be int or str")
    if not isinstance(value_epoch, (int, str)):
        raise TypeError("value_epoch must be int or str")

    map_env = Map(size=map_size, z_range=z_range)
    server = None
    if not headless:
        server = viser.ViserServer()
        print("Viser server started at http://localhost:8080")
        setup_camera_top_view(server)
        map_env.visualize_bounds(server)

    obstacles = map_env.generate_obstacles(
        server=server,
        num_obstacles=obstacle_count,
        min_size=obstacle_min_size,
        max_size=obstacle_max_size,
        seed=seed,
        color=obstacle_color,
        check_overlap=True,
        obstacle_type="mixed",
    )
    collision_checker = ObstacleCollisionChecker(obstacles)

    policy = _build_diffusion_policy(
        loadbase=loadbase,
        dataset=dataset,
        config=config,
        diffusion_loadpath=diffusion_loadpath,
        value_loadpath=value_loadpath,
        diffusion_epoch=diffusion_epoch,
        value_epoch=value_epoch,
        n_guide_steps=n_guide_steps,
        scale=scale,
        seed=seed,
    )

    condition = np.concatenate([start_state, goal_state]).astype(float)
    trajectory_candidates = _sample_trajectories(
        policy=policy,
        condition=condition,
        sample_batch_size=sample_batch_size,
        condition_key=condition_key,
    )

    selected = _select_collision_free_trajectory(
        trajectories=trajectory_candidates,
        collision_checker=collision_checker,
        bounds=np.asarray(map_env.get_bounds(), dtype=float),
        start_state=start_state,
        goal_state=goal_state,
        endpoint_tolerance=endpoint_tolerance,
        segment_resolution=segment_resolution,
    )
    if selected is None:
        print("No collision-free trajectory found in sampled batch.")
        return None

    print(f"Selected trajectory shape: {selected.shape}")
    print(f"Selected trajectory first/last: {selected[0]} -> {selected[-1]}")

    if not headless and server is not None:
        server.scene.add_line_segments(
            "/diffusion/trajectory",
            points=np.stack([selected[:-1], selected[1:]], axis=1),
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

        if save_image:

            @server.on_client_connect
            def handle_save(client: viser.ClientHandle) -> None:
                print("Saving image...")
                import time

                time.sleep(2)
                save_docs_image(client, "diffusion_trajectory_one_shot_example.png")
                print("Saved docs/images/diffusion_trajectory_one_shot_example.png")

        print("Press Ctrl+C to exit.")
        try:
            while True:
                import time

                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Shutting down server.")

    return selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate full trajectory from diffusion policy in one call."
    )
    parser.add_argument(
        "--run-config",
        type=str,
        default="config/diffusion_trajectory_one_shot.yaml",
        help="Run config path. Example: config/diffusion_trajectory_one_shot.yaml",
    )
    parser.add_argument("--headless", action="store_true", help="Run without viser visualizer.")
    parser.add_argument("--save-image", action="store_true", help="Save visualization image.")
    args = parser.parse_args()
    main(run_config_path=args.run_config, headless=args.headless, save_image=args.save_image)
