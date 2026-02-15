"""RRT with diffusion-guided sampling in a 3D environment."""

from __future__ import annotations

import argparse
import errno
import socket
import time
from pathlib import Path

import numpy as np
import viser

from planning.collision import ObstacleCollisionChecker
from planning.map import Map
from planning.sampling import RRT, DiffusionGuidedSampler, RRTConfig
from planning.visualization import RRTVisualizer, save_docs_image, setup_camera_top_view


def _can_bind_default_viser_port() -> bool:
    try:
        test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except OSError:
        return False

    try:
        test_socket.bind(("127.0.0.1", 8080))
        return True
    except OSError as exc:
        return exc.errno != errno.EACCES
    finally:
        test_socket.close()


def _expand_checkpoint_patterns(loadpath: str) -> str:
    """Replace config-style placeholders with wildcards for path discovery."""
    path_template = loadpath.removeprefix("f:")
    return (
        path_template.replace("{horizon}", "*")
        .replace("{n_diffusion_steps}", "*")
        .replace("{discount}", "*")
    )


def _find_matching_checkpoints(checkpoint_root: Path, loadpath: str) -> list[Path]:
    pattern = _expand_checkpoint_patterns(loadpath)
    return sorted(checkpoint_root.glob(pattern))


def _load_run_config(config_path: str) -> dict[str, object]:
    try:
        import yaml
    except Exception as exc:
        raise ImportError(
            "PyYAML is required to load run-config for rrt_diffuser_3d_example."
        ) from exc

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Run config not found: {path}")

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise TypeError(f"Run config must be a YAML mapping: {path}")
    return payload


def _ensure_vector3(values: object, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must be a 3D vector. Got shape {arr.shape}.")
    return arr


def _ensure_vector2(values: object, name: str) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.shape != (2,):
        raise ValueError(f"{name} must have exactly two values. Got shape {arr.shape}.")
    return float(arr[0]), float(arr[1])


def main(  # noqa: C901
    run_config_path: str = "config/rrt_diffuser_3d_example.yaml",
    save_image: bool = False,
    headless: bool = False,
) -> None:
    """Run 3D RRT planning with diffusion-guided sampler."""

    cfg = _load_run_config(run_config_path)
    print("=== RRT with Diffusion-Guided Sampler (3D) ===\n")
    print(f"Loaded run config: {run_config_path}\n")

    seed = int(cfg.get("seed", 42))
    loadbase = str(cfg["loadbase"])
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
    max_projection_retries = int(cfg.get("max_projection_retries", 128))
    obstacle_count = int(cfg.get("obstacle_count", 18))
    map_size = float(cfg.get("map_size", 24))
    z_range = _ensure_vector2(cfg.get("z_range", (0.5, 2.5)), "z_range")
    obstacle_min_size = float(cfg.get("obstacle_min_size", 0.5))
    obstacle_max_size = float(cfg.get("obstacle_max_size", 2.0))
    obstacle_color = cfg.get("obstacle_color", (200, 100, 50))
    start_state = _ensure_vector3(cfg.get("start_state", (8.0, 8.0, 2.0)), "start_state")
    goal_state = _ensure_vector3(cfg.get("goal_state", (-8.0, -8.0, 1.0)), "goal_state")

    rrt_defaults = cfg.get("rrt", {})
    if not isinstance(rrt_defaults, dict):
        raise TypeError(f"rrt config block must be a mapping, got {type(rrt_defaults)!r}")
    max_iterations = int(rrt_defaults.get("max_iterations", 4000))
    step_size = float(rrt_defaults.get("step_size", 0.5))
    goal_tolerance = float(rrt_defaults.get("goal_tolerance", 0.6))

    if not isinstance(diffusion_epoch, (int, str)):
        raise TypeError("diffusion_epoch must be int or string")
    if not isinstance(value_epoch, (int, str)):
        raise TypeError("value_epoch must be int or string")

    dataset_name = Path(dataset).stem
    checkpoint_root = Path(loadbase) / dataset_name
    if not checkpoint_root.exists():
        available_datasets = []
        loadbase_path = Path(loadbase)
        if loadbase_path.exists():
            available_datasets = [item.name for item in loadbase_path.iterdir() if item.is_dir()]
            suffix = (
                f"\n  Available dataset directories under {loadbase}: "
                f"{', '.join(available_datasets) if available_datasets else '<none>'}"
            )
        else:
            suffix = "\n  Also, this loadbase path does not exist yet."
        print(
            "Warning: checkpoint root not found for sampling config."
            f"\n  Expected: {checkpoint_root}"
            "\n  If this is your first run, provide matching loadbase/dataset in run config."
            f"{suffix}"
        )
    else:
        diffusion_matches = _find_matching_checkpoints(
            checkpoint_root=checkpoint_root, loadpath=diffusion_loadpath
        )
        value_matches = _find_matching_checkpoints(
            checkpoint_root=checkpoint_root, loadpath=value_loadpath
        )
        if not diffusion_matches:
            print(
                "Warning: no diffusion checkpoint directory matched for template "
                f"'{diffusion_loadpath}'. Checked under: {checkpoint_root}"
            )
        if not value_matches:
            print(
                "Warning: no value checkpoint directory matched for template "
                f"'{value_loadpath}'. Checked under: {checkpoint_root}"
            )

    # Optional Viser setup (disable in headless mode / restricted environments).
    server = None
    visualizer = None
    if headless:
        print("Running in headless mode (visualization disabled).")
    else:
        if not _can_bind_default_viser_port():
            print(
                "Cannot bind to local sockets in this environment; falling back to"
                " headless mode."
            )
            headless = True
        else:
            server = viser.ViserServer()
            print("Viser server started!")
            print("Open http://localhost:8080 in your browser.\n")
            setup_camera_top_view(server)

    # Create map.
    map_env = Map(size=map_size, z_range=z_range)
    print(f"Created map: {map_env}")
    print(f"Map bounds: {map_env.get_bounds()}\n")
    if server is not None:
        map_env.visualize_bounds(server)

    obstacles = map_env.generate_obstacles(
        server=server,
        num_obstacles=obstacle_count,
        min_size=obstacle_min_size,
        max_size=obstacle_max_size,
        seed=seed,
        color=tuple(obstacle_color),  # type: ignore[arg-type]
        check_overlap=True,
        obstacle_type="mixed",
    )
    print(f"Generated {len(obstacles)} obstacles\n")

    # Define start and goal.
    goal_condition = np.concatenate([start_state, goal_state]).astype(float)

    if not headless and server is not None:
        visualizer = RRTVisualizer(server)
        visualizer.visualize_start_goal(start_state, goal_state)

    # Create collision checker with generated obstacles.
    collision_checker = ObstacleCollisionChecker(map_env.obstacles)

    # Create RRT planner with diffuser-guided sampler.
    rrt = RRT(
        start_state=start_state,
        goal_state=goal_state,
        bounds=map_env.get_bounds(),
        collision_checker=collision_checker,
        config=RRTConfig(
            sampler=DiffusionGuidedSampler,
            seed=seed,
            max_iterations=max_iterations,
            step_size=step_size,
            goal_tolerance=goal_tolerance,
            sampler_kwargs={
                "loadbase": loadbase,
                "dataset": dataset,
                "config": config,
                "diffusion_loadpath": diffusion_loadpath,
                "value_loadpath": value_loadpath,
                "diffusion_epoch": diffusion_epoch,
                "value_epoch": value_epoch,
                "n_guide_steps": n_guide_steps,
                "scale": scale,
                "sample_batch_size": sample_batch_size,
                "max_projection_retries": max_projection_retries,
                "condition": goal_condition,
                "state_indices": (0, 1, 2),
            },
        ),
    )

    print("Planning with RRT + DiffusionGuidedSampler...")
    print(f"  Start: {start_state}")
    print(f"  Goal: {goal_state}")
    print(f"  Bounds: {map_env.get_bounds()}")
    print(f"  Max iterations: {rrt.max_iterations}")
    print(f"  Step size: {rrt.step_size}")
    print(f"  Goal tolerance: {rrt.goal_tolerance}\n")

    path = rrt.plan()

    if path is not None:
        print(f"\nPath found with {len(path)} waypoints!")
        print(f"Path length: {rrt.get_path_length():.2f}")
        print(f"Total nodes explored: {len(rrt.get_all_nodes())}\n")
        if visualizer is not None:
            visualizer.visualize_branches(
                rrt,
                success_color=(100, 150, 255),
                failure_color=(255, 100, 100),
                line_width=1.5,
            )
            print("\nVisualization complete!")
            print("Legend:")
            print("   Green sphere: Start")
            print("   Red sphere: Goal")
            print("   Blue lines: Explored branches")
            print("   Red lines: Failed branches")
    else:
        print("\nNo path found!")
        print("Try a different run config, sample budget, or checkpoint.")

    stats = rrt.get_stats()
    print("\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    if save_image:
        if server is None:
            print("Skipping save_image because headless mode is enabled.")
        else:

            @server.on_client_connect
            def handle_save(client: viser.ClientHandle) -> None:
                print("\nSaving image...")
                time.sleep(2)
                save_docs_image(client, "rrt_diffuser_3d_example.png")
                print("Image saved to docs/images/rrt_diffuser_3d_example.png")

    if headless:
        print("\nHeadless mode: exiting after planning.")
        return

    print("\nPress Ctrl+C to exit.")
    while True:
        try:
            time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nShutting down server.")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RRT + diffusion-guided sampler example")
    parser.add_argument(
        "--run-config",
        type=str,
        default="config/rrt_diffuser_3d_example.yaml",
        help=(
            "YAML config path used for every sampling/planning setting."
            " Example: config/rrt_diffuser_3d_example.yaml"
        ),
    )
    parser.add_argument("--save-image", action="store_true", help="Save documentation image")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Disable Viser/visualization and run planner headlessly",
    )
    args = parser.parse_args()

    main(
        run_config_path=args.run_config,
        save_image=args.save_image,
        headless=args.headless,
    )
