"""RRT* algorithm example with mixed obstacle types."""

import numpy as np
import viser

from planning.collision import ObstacleCollisionChecker
from planning.map import Map
from planning.sampling import (
    GoalBiasedSampler,  # , UniformSampler
    RRTStar,
    RRTStarConfig,
)
from planning.visualization import setup_camera_top_view
from planning.visualization.rrg_visualizer import RRGVisualizer


def main(seed: int = 42) -> None:
    """RRT* with mixed obstacle types and 3D visualization."""
    print("=== RRT* with Mixed Obstacles (Boxes & Spheres) ===\n")

    # Start Viser server
    server = viser.ViserServer()
    print("Viser server started!")
    print("Open http://localhost:8080 in your browser.\n")

    # Setup camera view
    setup_camera_top_view(server)

    # Create map
    map_env = Map(size=20, z_range=(0.5, 2.5))
    print(f"Created map: {map_env}")
    print(f"Map bounds: {map_env.get_bounds()}\n")

    # Visualize map bounds
    map_env.visualize_bounds(server)

    print("Generating mixed obstacles (boxes and spheres)...")
    obstacles = map_env.generate_obstacles(
        server=server,
        num_obstacles=15,
        min_size=0.5,
        max_size=2.5,
        seed=seed,
        color=(200, 100, 50),
        check_overlap=True,
        obstacle_type="mixed",
    )
    print(f"Generated {len(obstacles)} obstacles\n")

    # Define start and goal
    start_state = np.array([8.0, 8.0, 2.0])
    goal_state = np.array([-8.0, -8.0, 1.0])

    # Create visualizer
    visualizer = RRGVisualizer(server)

    # Add start and goal markers
    visualizer.visualize_start_goal(start_state, goal_state)

    # Create collision checker with map obstacles
    collision_checker = ObstacleCollisionChecker(map_env.obstacles)

    # Create RRT* planner
    rrt_star = RRTStar(
        start_state=start_state,
        goal_state=goal_state,
        bounds=map_env.get_bounds(),
        collision_checker=collision_checker,
        config=RRTStarConfig(
            sampler=GoalBiasedSampler,
            seed=seed,
            step_size=0.5,
            max_iterations=5000,
            radius_gain=1.0,
            goal_bias=0.05,
        ),
    )

    print("Planning with RRT*...")
    print(f"  Start: {start_state}")
    print(f"  Goal: {goal_state}")
    print(f"  Bounds: {map_env.get_bounds()}")
    print(f"  Max iterations: {rrt_star.max_iterations}")
    print(f"  Step size: {rrt_star.step_size}")
    print(f"  Goal tolerance: {rrt_star.goal_tolerance}")
    print(f"  Radius gain: {rrt_star.radius_gain}\n")

    # Run planner
    path = rrt_star.plan()

    if path is not None:
        print(f"\n✅ Path found with {len(path)} waypoints!")
        print(f"Path length: {rrt_star.get_path_length():.2f}")
        print(f"Total nodes in graph: {len(rrt_star.graph.nodes)}")
        print(f"Total edges in graph: {len(rrt_star.graph.edges)}\n")

        # Visualize the graph and the final path
        visualizer.visualize_graph(
            rrt_star,
            success_color=(100, 150, 255),  # Blue
            failure_color=(255, 100, 100),  # Red
            success_line_width=4,
            failure_line_width=4,
        )

        print("\nVisualization complete!")
        print("Legend:")
        print("  🟢 Green sphere: Start")
        print("  🔴 Red sphere: Goal")
        print("  🔵 Blue lines: Optimal path found by RRT*")
        print("  🔴 Red lines: Other edges in the graph")
        print("  📦 Orange boxes/spheres: Obstacles")

    else:
        print("\n❌ No path found!")
        print("Try increasing max_iterations or changing the seed.")
        # Visualize the graph even if no path is found
        visualizer.visualize_graph(rrt_star)

    # Statistics
    stats = rrt_star.get_stats()
    print("\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Keep server running
    print("\nPress Ctrl+C to exit.")
    while True:
        try:
            import time

            time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nShutting down server.")
            break


if __name__ == "__main__":
    main()
