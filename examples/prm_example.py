"""PRM (Probabilistic Roadmap Method) algorithm example."""

import numpy as np
import viser

from planning.collision import ObstacleCollisionChecker
from planning.map import Map
from planning.sampling.prm import PRM, PRMConfig
from planning.sampling.sampler import UniformSampler
from planning.visualization import setup_camera_top_view
from planning.visualization.rrg_visualizer import RRGVisualizer


def main(seed: int = 42) -> None:
    """PRM with mixed obstacle types and 3D visualization."""
    print("=== PRM (Probabilistic Roadmap Method) Example ===\n")

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

    # Create PRM planner
    prm = PRM(
        start_state=start_state,
        goal_state=goal_state,
        bounds=map_env.get_bounds(),
        collision_checker=collision_checker,
        config=PRMConfig(
            sampler=UniformSampler,
            seed=seed,
            step_size=0.1,
            sample_number=100,
            max_retries=5,
            radius=2.0,
            goal_tolerance=0.5,
        ),
    )

    print("Planning with PRM...")
    print(f"  Start: {start_state}")
    print(f"  Goal: {goal_state}")
    print(f"  Bounds: {map_env.get_bounds()}")
    print(f"  Sample number: {prm.sample_number}")
    print(f"  Max retries: {prm.max_retries}")
    print(f"  Connection radius: {prm.radius}")
    print(f"  Step size: {prm.step_size}")
    print(f"  Goal tolerance: {prm.goal_tolerance}\n")

    # Run planner
    path = prm.plan()

    if path is not None:
        print(f"\n✅ Path found with {len(path)} waypoints!")
        print(f"Total nodes in roadmap: {len(prm.graph.nodes)}")
        print(f"Total edges in roadmap: {len(prm.graph.edges)}\n")

        # Visualize the roadmap and the final path
        visualizer.visualize_graph(
            prm,
            success_color=(100, 150, 255),  # Blue for path
            failure_color=(255, 100, 100),  # Gray for roadmap
            success_line_width=5.0,
            failure_line_width=1.0,
        )

        print("\nVisualization complete!")
        print("Legend:")
        print("  🟢 Green sphere: Start")
        print("  🔴 Red sphere: Goal")
        print("  🔵 Blue lines: Final path (A* search result)")
        print("  ⚪ Gray lines: Roadmap edges")
        print("  📦 Orange boxes/spheres: Obstacles")
        print("\nNote: PRM builds a roadmap in preprocessing phase,")
        print("      then uses A* to find the shortest path in the roadmap.")

    else:
        print("\n❌ No path found!")
        print("Try increasing sample_number, max_retries, or connection radius.")
        # Visualize the roadmap even if no path is found
        visualizer.visualize_graph(prm)

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
