"""PRM* (Probabilistic Roadmap Method Star) algorithm example."""

import argparse

import numpy as np
import viser

from planning.collision import ObstacleCollisionChecker
from planning.map import Map
from planning.sampling.prm import PRMStar, PRMStarConfig
from planning.sampling.sampler import UniformSampler
from planning.visualization import save_docs_image, setup_camera_top_view
from planning.visualization.rrg_visualizer import RRGVisualizer


def main(seed: int = 42, save_image: bool = False) -> None:
    """PRM* with mixed obstacle types and 3D visualization."""
    print("=== PRM* (Probabilistic Roadmap Method Star) Example ===\n")

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

    # Create PRM* planner
    prm_star = PRMStar(
        start_state=start_state,
        goal_state=goal_state,
        bounds=map_env.get_bounds(),
        collision_checker=collision_checker,
        config=PRMStarConfig(
            sampler=UniformSampler,
            seed=seed,
            step_size=0.1,
            sample_number=300,
            radius_gain=5.0,
            max_retries=5,
            goal_tolerance=0.5,
        ),
    )

    print("Planning with PRM*...")
    print(f"  Start: {start_state}")
    print(f"  Goal: {goal_state}")
    print(f"  Bounds: {map_env.get_bounds()}")
    print(f"  Sample number: {prm_star.sample_number}")
    print(f"  Max retries: {prm_star.max_retries}")
    print(f"  Step size: {prm_star.step_size}")
    print(f"  Radius gain: {prm_star.radius_gain}")
    print(f"  Goal tolerance: {prm_star.goal_tolerance}\n")

    # Run planner
    path = prm_star.plan()

    if path is not None:
        print(f"\n✅ Path found with {len(path)} waypoints!")
        print(f"Total nodes in roadmap: {len(prm_star.graph.nodes)}")
        print(f"Total edges in roadmap: {len(prm_star.graph.edges)}\n")

        # Get stats
        stats = prm_star.get_stats()
        if stats["path_length"] is not None:
            print(f"Path length: {stats['path_length']:.2f}")

        # Visualize the roadmap and the final path
        visualizer.visualize_graph(
            prm_star,
            success_color=(100, 150, 255),  # Blue for path
            failure_color=(255, 100, 100),  # Red for roadmap
            success_line_width=5.0,
            failure_line_width=1.0,
        )

        print("\nVisualization complete!")
        print("Legend:")
        print("  🟢 Green sphere: Start")
        print("  🔴 Red sphere: Goal")
        print("  🔵 Blue lines: Final path (A* search result)")
        print("  🔴 Red lines: Roadmap edges")
        print("  📦 Orange boxes/spheres: Obstacles")
        print("\nNote: PRM* uses dynamic radius calculation based on log(n)/n")
        print("      for asymptotic optimality, ensuring connection radius")
        print("      decreases as the number of samples increases.")

    else:
        print("\n❌ No path found!")
        print("Try increasing sample_number, max_retries, or radius_gain.")
        # Visualize the roadmap even if no path is found
        visualizer.visualize_graph(prm_star)

    # Save image if requested
    if save_image:

        @server.on_client_connect
        def handle_save(client: viser.ClientHandle) -> None:
            """Save documentation image after client connects."""
            import time

            print("\n📸 Saving image...")
            time.sleep(2)  # Wait for rendering
            save_docs_image(client, "prm_star_example.png")
            print("✅ Image saved to docs/images/prm_star_example.png")

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
    parser = argparse.ArgumentParser(description="PRM* algorithm example")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-image", action="store_true", help="Save documentation image")
    args = parser.parse_args()

    main(seed=args.seed, save_image=args.save_image)
