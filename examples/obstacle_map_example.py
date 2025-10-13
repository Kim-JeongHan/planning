"""Random obstacle map generation example."""

import viser

from planning.map import BoxObstacle, Map, SphereObstacle


def main() -> None:
    """Example of generating random obstacles using Map class."""
    # Start Viser server
    server = viser.ViserServer()
    print("Viser server started!")
    print("Open http://localhost:8080 in your browser.\n")

    # Create map
    map_env = Map(size=20, z_range=(0.5, 2.5))
    print(f"Created map: {map_env}")
    print(f"Map bounds: {map_env.get_bounds()}\n")

    # Visualize map boundaries
    map_env.visualize_bounds(server)

    # Generate random obstacles
    obstacles = map_env.generate_obstacles(
        server=server,
        num_obstacles=15,
        min_size=0.5,
        max_size=3.0,
        seed=42,
        color=(200, 100, 50),
        check_overlap=True,
        obstacle_type="box",  # Generate only box obstacles for this example
    )

    print(f"Number of obstacles generated: {len(obstacles)}")
    print(f"Total obstacles in map: {len(map_env.obstacles)}\n")
    print("Obstacle information:")
    for i, obs in enumerate(obstacles):
        if isinstance(obs, BoxObstacle):
            print(f"  📦 Box {i+1}: position={obs.position}, size={obs.size}")
        elif isinstance(obs, SphereObstacle):
            print(f"  ⚪ Sphere {i+1}: center={obs.center}, radius={obs.radius}")

    # Add coordinate frame
    server.scene.add_frame("/axes", wxyz=(1, 0, 0, 0), position=(0, 0, 0), axes_length=2.0)

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
