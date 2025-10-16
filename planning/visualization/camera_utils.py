"""Camera utilities for Viser visualization."""

import viser


def setup_camera_top_view(
    server: viser.ViserServer,
    distance: float = 18,
    look_at: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    """Setup camera for top-down view.

    Args:
        server: Viser server instance
        distance: Distance from the center point
        look_at: Point to look at (x, y, z)
    """

    @server.on_client_connect
    def handle_client(client: viser.ClientHandle) -> None:
        """Handle client connection and setup camera."""
        client.camera.position = (0.0, 0.0, distance)
        client.camera.look_at = look_at
        client.camera.up_direction = (0.0, 0.0, 1.0)


def setup_camera_side_view(
    server: viser.ViserServer,
    distance: float = 20.0,
    look_at: tuple[float, float, float] = (0.0, 0.0, 0.0),
    angle: float = 45.0,
) -> None:
    """Setup camera for side view at an angle.

    Args:
        server: Viser server instance
        distance: Distance from the center point
        look_at: Point to look at (x, y, z)
        angle: Viewing angle in degrees (0 = side, 90 = top)
    """
    import numpy as np

    # Convert angle to radians
    angle_rad = np.deg2rad(angle)

    # Calculate camera position
    x = distance * np.cos(angle_rad)
    z = distance * np.sin(angle_rad)

    @server.on_client_connect
    def handle_client(client: viser.ClientHandle) -> None:
        """Handle client connection and setup camera."""
        client.camera.position = (x, 0.0, z)
        client.camera.look_at = look_at
        client.camera.up_direction = (0.0, 0.0, 1.0)


def setup_camera_isometric_view(
    server: viser.ViserServer,
    distance: float = 25.0,
    look_at: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    """Setup camera for isometric view (45° angle, 45° rotation).

    Args:
        server: Viser server instance
        distance: Distance from the center point
        look_at: Point to look at (x, y, z)
    """
    import numpy as np

    # Isometric view: 45° elevation, 45° azimuth
    elevation = np.deg2rad(45)
    azimuth = np.deg2rad(45)

    x = distance * np.cos(elevation) * np.cos(azimuth)
    y = distance * np.cos(elevation) * np.sin(azimuth)
    z = distance * np.sin(elevation)

    @server.on_client_connect
    def handle_client(client: viser.ClientHandle) -> None:
        """Handle client connection and setup camera."""
        client.camera.position = (x, y, z)
        client.camera.look_at = look_at
        client.camera.up_direction = (0.0, 0.0, 1.0)
