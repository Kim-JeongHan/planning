"""RRT-Connect algorithm example with obstacles."""

import numpy as np
import viser

from planning.map import Map
from planning.sampling import ObstacleCollisionChecker, RRTConnect, RRTConnectConfig
from planning.visualization import RRTVisualizer


def main(seed: int = 42) -> None:
    """Example of RRT-Connect path planning with obstacles."""
    # Start Viser server
    server = viser.ViserServer()
    print("🚀 Viser server started!")
    print("📱 Open http://localhost:8080 in your browser.\n")

    # Create map
    map_env = Map(size=20, z_range=(0.5, 2.5))
    print(f"🗺️  Created map: {map_env}")
    print(f"📏 Map bounds: {map_env.get_bounds()}\n")

    # Visualize map boundaries
    map_env.visualize_bounds(server)

    # Generate random obstacles (mixed box and sphere)
    print("🎲 Generating random obstacles...")
    obstacles = map_env.generate_obstacles(
        server=server,
        num_obstacles=40,
        min_size=0.5,
        max_size=2.5,
        seed=seed,
        color=(200, 100, 50),
        check_overlap=True,
        obstacle_type="box",
    )
    print(f"✅ Generated {len(obstacles)} obstacles\n")

    # Define start and goal states
    start_state = np.array([-8.0, -8.0, 1.0])
    goal_state = np.array([8.0, 8.0, 2.0])

    print(f"🎯 Start: {start_state}")
    print(f"🏁 Goal:  {goal_state}\n")

    # Create RRT-Connect planner
    print("🤖 Initializing RRT-Connect planner...")
    rrt_connect = RRTConnect(
        start_state=start_state,
        goal_state=goal_state,
        bounds=map_env.get_bounds(),
        collision_checker=ObstacleCollisionChecker(map_env.obstacles),
        config=RRTConnectConfig(
            max_iterations=5000,
            seed=seed,
        ),
    )

    # Plan path
    print("🔍 Planning path with RRT-Connect...\n")
    path = rrt_connect.plan()

    # Create visualizer
    visualizer = RRTVisualizer(server)
    visualizer.visualize_start_goal(start_state, goal_state)

    if path is not None:
        print(f"\n✅ Path found with {len(path)} waypoints!")
        print(f"Path length: {rrt_connect.get_path_length():.2f}")
        print(f"Total nodes explored: {len(rrt_connect.get_all_nodes())}\n")

        visualizer.visualize_branches(
            rrt_connect,  # Pass the planner directly
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
    stats = rrt_connect.get_stats()
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
