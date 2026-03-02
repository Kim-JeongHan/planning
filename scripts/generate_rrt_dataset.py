"""Generate a trajectory dataset by running planner repeatedly in a fixed environment."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import TypeVar, cast

import numpy as np
import yaml  # type: ignore[import-untyped]
from tqdm import tqdm

from planning.collision import ObstacleCollisionChecker
from planning.map import Map
from planning.sampling import RRT, RRTConfig

_T = TypeVar("_T")


def _sample_free_state(
    rng: np.random.Generator,
    map_env: Map,
    checker: ObstacleCollisionChecker,
    bounds: list[tuple[float, float]],
) -> np.ndarray:
    for _ in range(1_000):
        state = np.array(
            [
                rng.uniform(low=bounds[0][0], high=bounds[0][1]),
                rng.uniform(low=bounds[1][0], high=bounds[1][1]),
                rng.uniform(low=map_env.z_min, high=map_env.z_max),
            ],
            dtype=float,
        )
        if map_env.is_valid_state(state) and checker.is_collision_free(state):
            return state
    raise RuntimeError("Failed to sample collision-free state after many attempts.")


def _resample_path_to_horizon(path: np.ndarray, horizon: int) -> np.ndarray:
    if path.ndim != 2 or path.shape[1] != 3:
        raise ValueError("path must be [N, 3]")

    if len(path) == horizon:
        return path.astype(np.float32)
    if len(path) == 1:
        raise ValueError("cannot resample a single-point path")

    t_src = np.linspace(0.0, 1.0, num=len(path), endpoint=True)
    t_tgt = np.linspace(0.0, 1.0, num=horizon, endpoint=True)
    out = np.empty((horizon, path.shape[1]), dtype=np.float32)
    for dim in range(path.shape[1]):
        out[:, dim] = np.interp(t_tgt, t_src, path[:, dim]).astype(np.float32)
    return out


def build_dataset(
    *,
    output: Path,
    dataset_size: int,
    horizon: int,
    map_size: float,
    obstacle_count: int,
    seed: int,
    max_iterations: int,
    step_size: float,
    goal_tolerance: float,
    quiet: bool,
) -> int:
    rng = np.random.default_rng(seed)
    map_env = Map(size=map_size, z_range=(0.5, 2.5))
    map_env.generate_obstacles(
        server=None,
        num_obstacles=obstacle_count,
        min_size=0.5,
        max_size=2.0,
        seed=seed,
        check_overlap=True,
        obstacle_type="mixed",
    )
    checker = ObstacleCollisionChecker(map_env.obstacles)

    bounds = map_env.get_bounds()
    trajectories: list[np.ndarray] = []

    for _ in tqdm(range(dataset_size), desc="Generating trajectories", disable=quiet):
        start_state = _sample_free_state(rng, map_env, checker, bounds)
        goal_state = _sample_free_state(rng, map_env, checker, bounds)
        while np.linalg.norm(start_state - goal_state) < 1.5:
            goal_state = _sample_free_state(rng, map_env, checker, bounds)

        planner = RRT(
            start_state=start_state,
            goal_state=goal_state,
            bounds=bounds,
            collision_checker=checker,
            config=RRTConfig(
                seed=int(rng.integers(0, 2**31 - 1)),
                max_iterations=max_iterations,
                step_size=step_size,
                goal_tolerance=goal_tolerance,
            ),
        )
        path = planner.plan()
        if path is None:
            continue

        path_states = np.asarray([node.state for node in path], dtype=np.float32)
        if len(path_states) < 2:
            continue

        trajectories.append(_resample_path_to_horizon(path_states, horizon))

    if not trajectories:
        raise RuntimeError("No valid trajectories were generated. Relax constraints.")

    observations = np.stack(trajectories, axis=0)
    np.savez(output, observations=observations)
    return len(trajectories)


def _load_config(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Dataset generation config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return payload


def _select_config_value(
    cli_value: _T | None,
    config: Mapping[str, object],
    key: str,
    default: _T,
) -> _T:
    if cli_value is not None:
        return cli_value
    value = config.get(key, default)
    return cast(_T, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate planner trajectories for local diffuser training."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML configuration path (overridden by CLI values)",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output NPZ path")
    parser.add_argument("--dataset-size", type=int, default=None, help="Number of attempts")
    parser.add_argument("--horizon", type=int, default=None, help="Trajectory length in training")
    parser.add_argument("--map-size", type=float, default=None, help="Map square size")
    parser.add_argument("--obstacle-count", type=int, default=None, help="Random obstacle count")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed")
    parser.add_argument(
        "--quiet",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Disable progress bars while generating trajectories",
    )
    parser.add_argument("--max-iterations", type=int, default=None, help="RRT max iterations")
    parser.add_argument("--step-size", type=float, default=None, help="RRT step size")
    parser.add_argument("--goal-tolerance", type=float, default=None, help="RRT goal tolerance")
    parsed = parser.parse_args()

    config = _load_config(parsed.config)
    return argparse.Namespace(
        output=parsed.output,
        dataset_size=_select_config_value(parsed.dataset_size, config, "dataset_size", 200),
        horizon=_select_config_value(parsed.horizon, config, "horizon", 16),
        map_size=_select_config_value(parsed.map_size, config, "map_size", 24.0),
        obstacle_count=_select_config_value(parsed.obstacle_count, config, "obstacle_count", 18),
        seed=_select_config_value(parsed.seed, config, "seed", 42),
        max_iterations=_select_config_value(parsed.max_iterations, config, "max_iterations", 4000),
        step_size=_select_config_value(parsed.step_size, config, "step_size", 0.5),
        goal_tolerance=_select_config_value(parsed.goal_tolerance, config, "goal_tolerance", 0.6),
        quiet=_select_config_value(parsed.quiet, config, "quiet", False),
        config=parsed.config,
    )


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = build_dataset(
        output=args.output,
        dataset_size=args.dataset_size,
        horizon=args.horizon,
        map_size=args.map_size,
        obstacle_count=args.obstacle_count,
        seed=args.seed,
        max_iterations=args.max_iterations,
        step_size=args.step_size,
        goal_tolerance=args.goal_tolerance,
        quiet=args.quiet,
    )
    print(f"Saved {count} trajectories -> {args.output}")


if __name__ == "__main__":
    main()
