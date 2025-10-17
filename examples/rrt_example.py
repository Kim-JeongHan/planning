"""RRT algorithm example with mixed obstacle types."""

import argparse

import numpy as np
import viser

from planning.collision import ObstacleCollisionChecker
from planning.map import BoxObstacle, Map, SphereObstacle
from planning.sampling import RRT, RRTConfig
from planning.visualization import RRTVisualizer, save_docs_image, setup_camera_top_view


def main(seed: int = 42, save_image: bool = False) -> None:
    """RRT with mixed obstacle types and 3D visualization."""

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
        obstacle_type="mixed",  # <-- Mixed obstacles!
    )
    print(f"Generated {len(obstacles)} obstacles\n")

    box_count = sum(1 for obs in obstacles if isinstance(obs, BoxObstacle))
    sphere_count = sum(1 for obs in obstacles if isinstance(obs, SphereObstacle))
    print(f"  📦 Box obstacles: {box_count}")
    print(f"  ⚪ Sphere obstacles: {sphere_count}\n")

    # Define start and goal
    start_state = np.array([8.0, 8.0, 2.0])
    goal_state = np.array([-8.0, -8.0, 1.0])

    # Create visualizer
    visualizer = RRTVisualizer(server)

    # Add start and goal markers
    visualizer.visualize_start_goal(start_state, goal_state)

    # Create collision checker with map obstacles
    collision_checker = ObstacleCollisionChecker(map_env.obstacles)

    # Create RRT planner
    rrt = RRT(
        start_state=start_state,
        goal_state=goal_state,
        bounds=map_env.get_bounds(),
        collision_checker=collision_checker,
        config=RRTConfig(seed=seed, max_iterations=5000),
    )

    print("Planning with RRT...")
    print(f"  Start: {start_state}")
    print(f"  Goal: {goal_state}")
    print(f"  Bounds: {map_env.get_bounds()}")
    print(f"  Max iterations: {rrt.max_iterations}")
    print(f"  Step size: {rrt.step_size}")
    print(f"  Goal tolerance: {rrt.goal_tolerance}\n")

    # Run planner
    path = rrt.plan()

    if path is not None:
        print(f"\n✅ Path found with {len(path)} waypoints!")
        print(f"Path length: {rrt.get_path_length():.2f}")
        print(f"Total nodes explored: {len(rrt.get_all_nodes())}\n")

        # Visualize all paths (success: blue, failure: red)
        visualizer.visualize_branches(
            rrt,  # Pass the planner directly
            success_color=(100, 150, 255),  # Blue
            failure_color=(255, 100, 100),  # Red
            line_width=1.5,
        )

        print("\nVisualization complete!")
        print("Legend:")
        print("  🔵 Blue paths: Successful (led to goal)")
        print("  🔴 Red paths: Failed (dead ends)")
        print("  🟢 Green sphere: Start")
        print("  🔴 Red sphere: Goal")
        print("  📦 Orange boxes: Box obstacles")
        print("  ⚪ Orange spheres: Sphere obstacles")

    else:
        print("\n❌ No path found!")
        print("Try increasing max_iterations or decreasing obstacle count.")

    # Statistics
    stats = rrt.get_stats()
    print("\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Save image if requested
    if save_image:

        @server.on_client_connect
        def handle_save(client: viser.ClientHandle) -> None:
            """Save documentation image after client connects."""
            print("\n📸 Saving image...")
            time.sleep(2)  # Wait for rendering
            save_docs_image(client, "rrt_example.png")
            print("✅ Image saved to docs/images/rrt_example.png")

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
    parser = argparse.ArgumentParser(description="RRT algorithm example")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-image", action="store_true", help="Save documentation image")
    args = parser.parse_args()

    main(seed=args.seed, save_image=args.save_image)
