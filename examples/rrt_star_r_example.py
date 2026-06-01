"""Terrain-metric sampling planner example."""

from __future__ import annotations

import argparse
import time
from itertools import pairwise

import numpy as np
import viser

from planning.map import (
    MountainTerrain,
    TerrainPlan,
    TerrainRiemannianSpace,
    create_random_start_goal,
)
from planning.sampling import (
    RRG,
    PRMStar,
    PRMStarConfig,
    RRGConfig,
    RRTConnect,
    RRTConnectConfig,
    RRTStar,
    RRTStarConfig,
)
from planning.visualization import save_docs_image, setup_camera_isometric_view

PlannerName = str

PLANNER_CHOICES = ("rrt-connect", "rrg", "rrt-star", "prm-star")
PLANNER_LABELS = {
    "rrt-connect": "RRT-Connect",
    "rrg": "RRG",
    "rrt-star": "RRT*",
    "prm-star": "PRM*",
}
PLANNER_IMAGE_NAMES = {
    "rrt-connect": "rrt_star_r_rrt_connect_example.png",
    "rrg": "rrt_star_r_rrg_example.png",
    "rrt-star": "rrt_star_r_rrt_star_example.png",
    "prm-star": "rrt_star_r_prm_star_example.png",
}


def plan_terrain_path(
    terrain: MountainTerrain,
    start: np.ndarray | tuple[float, float] | None = None,
    goal: np.ndarray | tuple[float, float] | None = None,
    planner_name: PlannerName = "rrt-star",
    seed: int = 42,
    max_iterations: int = 600,
    sample_count: int = 500,
) -> TerrainPlan:
    """Plan a 2D path whose edge costs follow the terrain surface metric."""
    bounds = terrain.bounds

    if start is None and goal is None:
        start, goal = create_random_start_goal(bounds, seed)
    else:
        rng = np.random.default_rng(seed)
        lower = np.array([bound[0] for bound in bounds], dtype=float)
        upper = np.array([bound[1] for bound in bounds], dtype=float)
        if start is None:
            start = rng.uniform(lower, upper)
        if goal is None:
            goal = rng.uniform(lower, upper)

    start_state = np.array(start, dtype=float)
    goal_state = np.array(goal, dtype=float)
    space = TerrainRiemannianSpace(terrain)
    planner = create_planner(
        planner_name=planner_name,
        start_state=start_state,
        goal_state=goal_state,
        bounds=bounds,
        space=space,
        seed=seed,
        max_iterations=max_iterations,
        sample_count=sample_count,
    )
    path_nodes = planner.plan()
    if path_nodes is None:
        raise RuntimeError(
            f"{PLANNER_LABELS[planner_name]} failed: nodes={len(planner.get_all_nodes())}"
        )

    path_edge_states = [
        planner.graph.edge_states(from_node, to_node) for from_node, to_node in pairwise(path_nodes)
    ]
    graph_edge_states = collect_graph_edge_states(planner)
    sampled_nodes = np.array([node.state for node in planner.get_all_nodes()], dtype=float)

    return TerrainPlan(
        start=start_state,
        goal=goal_state,
        path=np.array([node.state for node in path_nodes], dtype=float),
        path_edge_states=path_edge_states,
        sampled_nodes=sampled_nodes,
        graph_edge_states=graph_edge_states,
        path_length=planner.get_path_length(),
    )


def create_planner(
    planner_name: PlannerName,
    start_state: np.ndarray,
    goal_state: np.ndarray,
    bounds: list[tuple[float, float]],
    space: TerrainRiemannianSpace,
    seed: int,
    max_iterations: int,
    sample_count: int,
) -> RRTConnect | RRG | RRTStar | PRMStar:
    """Create a terrain-metric planner by public planner name."""
    if planner_name == "rrt-connect":
        return RRTConnect(
            start_state=start_state,
            goal_state=goal_state,
            bounds=bounds,
            config=RRTConnectConfig(
                space=space,
                max_iterations=max_iterations,
                step_size=0.55,
                goal_tolerance=0.35,
                seed=seed,
            ),
        )
    if planner_name == "rrg":
        return RRG(
            start_state=start_state,
            goal_state=goal_state,
            bounds=bounds,
            config=RRGConfig(
                space=space,
                max_iterations=max_iterations,
                step_size=0.55,
                goal_tolerance=0.35,
                goal_bias=0.08,
                radius_gain=8.0,
                seed=seed,
            ),
        )
    if planner_name == "rrt-star":
        return RRTStar(
            start_state=start_state,
            goal_state=goal_state,
            bounds=bounds,
            config=RRTStarConfig(
                space=space,
                max_iterations=max_iterations,
                step_size=0.55,
                goal_tolerance=0.35,
                goal_bias=0.08,
                radius_gain=8.0,
                return_first_solution=False,
                seed=seed,
            ),
        )
    if planner_name == "prm-star":
        return PRMStar(
            start_state=start_state,
            goal_state=goal_state,
            bounds=bounds,
            config=PRMStarConfig(
                space=space,
                sample_number=sample_count,
                max_retries=1,
                step_size=0.55,
                goal_tolerance=0.35,
                goal_bias=0.08,
                radius_gain=8.0,
                seed=seed,
            ),
        )
    raise ValueError(f"Unsupported planner: {planner_name}")


def collect_graph_edge_states(planner: RRTConnect | RRG | RRTStar | PRMStar) -> list[np.ndarray]:
    """Return local edge states for graph/tree visualization."""
    if planner.graph.edges:
        return [planner.graph.edge_states(edge.node1, edge.node2) for edge in planner.graph.edges]
    return [
        planner.graph.edge_states(node.parent, node)
        for node in planner.get_all_nodes()
        if node.parent is not None
    ]


def edge_paths_to_surface_segments(
    terrain: MountainTerrain,
    edge_paths: list[np.ndarray],
    z_offset: float,
) -> np.ndarray:
    """Project local edge paths to 3D line segments on the terrain surface."""
    segments = []
    for edge_path in edge_paths:
        points = terrain.states_to_surface_points(edge_path, z_offset=z_offset)
        if len(points) >= 2:
            segments.append(np.stack([points[:-1], points[1:]], axis=1))
    if not segments:
        return np.empty((0, 2, 3), dtype=float)
    return np.concatenate(segments, axis=0)


def visualize_plan(
    server: viser.ViserServer,
    terrain: MountainTerrain,
    plan: TerrainPlan,
) -> None:
    """Show the terrain, sampled states, and final path in Viser."""
    path_points = terrain.states_to_surface_points(plan.path, z_offset=0.18)
    sampled_points = terrain.states_to_surface_points(plan.sampled_nodes, z_offset=0.08)
    graph_segments = edge_paths_to_surface_segments(
        terrain,
        plan.graph_edge_states,
        z_offset=0.07,
    )
    path_segments = edge_paths_to_surface_segments(
        terrain,
        plan.path_edge_states,
        z_offset=0.18,
    )
    start_point = terrain.states_to_surface_points(plan.start.reshape(1, 2), z_offset=0.35)[0]
    goal_point = terrain.states_to_surface_points(plan.goal.reshape(1, 2), z_offset=0.35)[0]

    server.scene.set_up_direction("+z")
    server.scene.add_grid(
        "/ground_grid",
        width=terrain.world_size,
        height=terrain.world_size,
        plane="xy",
        cell_size=1.0,
        section_size=2.0,
        cell_color=(120, 135, 120),
        section_color=(85, 95, 85),
        position=(0.0, 0.0, -0.03),
    )
    server.scene.add_mesh_simple(
        "/mountain/terrain",
        vertices=terrain.vertices,
        faces=terrain.faces,
        color=(88, 128, 72),
        flat_shading=False,
        side="double",
    )

    if len(sampled_points) > 0:
        server.scene.add_point_cloud(
            "/planner/sampled_nodes",
            points=sampled_points,
            colors=(70, 130, 230),
            point_size=0.045,
            point_shape="circle",
        )

    if len(graph_segments) > 0:
        server.scene.add_line_segments(
            "/planner/graph_edges",
            points=graph_segments,
            colors=(80, 120, 220),
            line_width=1.0,
        )

    if len(path_segments) > 0:
        server.scene.add_line_segments(
            "/planner/path",
            points=path_segments,
            colors=(255, 215, 40),
            line_width=4.0,
        )

    server.scene.add_point_cloud(
        "/planner/waypoints",
        points=path_points,
        colors=(255, 245, 120),
        point_size=0.09,
        point_shape="circle",
    )
    server.scene.add_icosphere(
        "/planner/start",
        radius=0.18,
        color=(40, 220, 100),
        position=tuple(start_point),
    )
    server.scene.add_icosphere(
        "/planner/goal",
        radius=0.18,
        color=(235, 70, 70),
        position=tuple(goal_point),
    )


def main(
    planner_name: PlannerName = "rrt-star",
    seed: int = 42,
    max_iterations: int = 600,
    sample_count: int = 500,
    save_image: bool = False,
    show: bool = True,
) -> None:
    """Run a metric-aware sampling planner with terrain metric visualization."""
    print("=== RRT*-R Terrain-Metric Planning Example ===\n")
    planner_label = PLANNER_LABELS[planner_name]

    server = None
    if show:
        # Start Viser server
        server = viser.ViserServer()
        print("Viser server started!")
        print("Open http://localhost:8080 in your browser.\n")

        # Setup camera view
        setup_camera_isometric_view(
            server,
            distance=17.0,
            look_at=(0.0, 0.0, 1.2),
            elevation_deg=34.0,
            azimuth_deg=-35.0,
        )

    # Create terrain
    terrain = MountainTerrain()
    print("Created mountain terrain")
    print(f"Terrain bounds: {terrain.bounds}")
    print(f"Grid size: {terrain.grid_size} x {terrain.grid_size}\n")

    # Define start and goal
    start_state = np.array([-5.5, -5.5])
    goal_state = np.array([5.5, 5.5])

    print(f"Planning with {planner_label} terrain metric...")
    print(f"  Start: {start_state}")
    print(f"  Goal: {goal_state}")
    print(f"  Bounds: {terrain.bounds}")
    print(f"  Planner: {planner_label}")
    print(f"  Max iterations: {max_iterations}")
    print(f"  PRM* samples: {sample_count}")
    print("  Step size: 0.55")
    print("  Goal tolerance: 0.35")
    print("  Radius gain: 8.0\n")

    try:
        plan = plan_terrain_path(
            terrain,
            start=start_state,
            goal=goal_state,
            planner_name=planner_name,
            seed=seed,
            max_iterations=max_iterations,
            sample_count=sample_count,
        )
    except RuntimeError as exc:
        print("\n No path found!")
        print(str(exc))
        print("Try increasing --iterations or changing --seed.")
        return

    print(f"\n Path found with {len(plan.path)} waypoints!")
    print(f"Path length: {plan.path_length:.2f}")
    print(f"Total nodes in graph: {len(plan.sampled_nodes)}")
    print(f"Total edges in graph: {len(plan.graph_edge_states)}\n")

    if show and server is not None:
        # Visualize the terrain, full graph, and final path
        visualize_plan(server, terrain, plan)

        print("\nVisualization complete!")
        print("Legend:")
        print("   Green sphere: Start")
        print("   Red sphere: Goal")
        print("   Yellow lines: Final terrain-metric path")
        print("   Yellow points: Final path waypoints")
        print(f"   Blue lines: {planner_label} graph edges")
        print("   Blue points: Sampled graph nodes")

    # Statistics
    stats = {
        "num_nodes": len(plan.sampled_nodes),
        "num_edges": len(plan.graph_edge_states),
        "path_length": plan.path_length,
        "path_nodes": len(plan.path),
        "path_edge_segments": sum(
            max(0, len(edge_path) - 1) for edge_path in plan.path_edge_states
        ),
    }
    print("\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    if not show:
        return

    assert server is not None

    # Save image if requested
    if save_image:

        @server.on_client_connect
        def handle_save(client: viser.ClientHandle) -> None:
            """Save documentation image after client connects."""
            print("\n Saving image...")
            time.sleep(2)  # Wait for rendering
            image_name = PLANNER_IMAGE_NAMES[planner_name]
            save_docs_image(client, image_name)
            print(f" Image saved to docs/images/{image_name}")

    # Keep server running
    print("\nPress Ctrl+C to exit.")
    while True:
        try:
            time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nShutting down server.")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Metric-aware terrain planning example")
    parser.add_argument(
        "--planner",
        choices=PLANNER_CHOICES,
        default="rrt-star",
        help="Planner to run with the terrain metric",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--iterations", type=int, default=600, help="Planner iteration budget")
    parser.add_argument("--samples", type=int, default=500, help="PRM* sample budget")
    parser.add_argument("--save-image", action="store_true", help="Save documentation image")
    parser.add_argument("--no-show", action="store_true", help="Plan without starting Viser")
    args = parser.parse_args()
    main(
        planner_name=args.planner,
        seed=args.seed,
        max_iterations=args.iterations,
        sample_count=args.samples,
        save_image=args.save_image,
        show=not args.no_show,
    )
