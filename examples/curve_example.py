"""3D curve drawing example using Viser."""

import numpy as np
import viser


def main():
    """Example of drawing 3D spiral curves."""
    # Start Viser server
    server = viser.ViserServer()
    print("Viser server started!")
    print("Open http://localhost:8080 in your browser.")
    
    # Generate spiral curve
    t = np.linspace(0, 4 * np.pi, 200)
    x = np.cos(t)
    y = np.sin(t)
    z = t / (4 * np.pi) * 2  # Rise to height 2
    
    # Combine points into (N, 3) shape
    points = np.stack([x, y, z], axis=1)
    
    # Draw curve by connecting line segments
    for i in range(len(points) - 1):
        server.scene.add_spline_catmull_rom(
            f"/spiral/segment_{i}",
            positions=points[i:i+2],
            color=(0, 255, 100),  # Green
            line_width=3.0,
        )
    
    # Additional curve: circular curve
    circle_t = np.linspace(0, 2 * np.pi, 100)
    circle_x = 2 * np.cos(circle_t)
    circle_y = 2 * np.sin(circle_t)
    circle_z = np.ones_like(circle_t) * 1.0  # Draw circle on z=1 plane
    
    circle_points = np.stack([circle_x, circle_y, circle_z], axis=1)
    
    # Draw circular curve
    server.scene.add_spline_catmull_rom(
        "/circle",
        positions=circle_points,
        color=(255, 100, 0),  # Orange
        line_width=2.0,
        closed=True,  # Close the curve
    )
    
    # Add coordinate frame
    server.scene.add_frame("/axes", wxyz=(1, 0, 0, 0), position=(0, 0, 0), axes_length=1.0)
    
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
