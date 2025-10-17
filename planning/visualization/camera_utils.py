"""Camera utilities for Viser visualization."""

import imageio.v3 as iio
import numpy as np
import viser
from PIL import Image

from ..path import DOC_IMAGES_DIR


def setup_camera_top_view(
    server: viser.ViserServer,
    distance: float = 17,
    look_at: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    """Setup camera for top-down view.

    Args:
        server: Viser server instance
        distance: Distance from the center point
        look_at: Point to look at (x, y, z)
    """

    @server.on_client_connect
    def handle_setup_camera(client: viser.ClientHandle) -> None:
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
    def handle_setup_camera(client: viser.ClientHandle) -> None:
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
    def handle_setup_camera(client: viser.ClientHandle) -> None:
        """Handle client connection and setup camera."""
        client.camera.position = (x, y, z)
        client.camera.look_at = look_at
        client.camera.up_direction = (0.0, 0.0, 1.0)


def capture_camera_view(
    server: viser.ViserServer,
    filename: str,
) -> None:
    """Capture a camera view and save it as an image."""

    @server.on_client_connect
    def capture_image(client: viser.ClientHandle) -> None:
        """Capture image and save it as an image."""
        image = client.camera.get_render(height=720, width=1280)
        iio.imwrite(filename, image, extension=".png")


def get_camera_view_image(
    client: viser.ClientHandle,
    height: int = 720,
    width: int = 1280,
) -> np.ndarray:
    """Capture a camera view and return it as an image array.

    Args:
        client: Viser client handle
        height: Image height in pixels
        width: Image width in pixels

    Returns:
        Image array (H, W, C)
    """
    return client.camera.get_render(height=height, width=width)


def remove_white_background_from_array(
    image_array: np.ndarray, threshold: int = 240
) -> Image.Image:
    """Remove white background from image array.

    Args:
        image_array: Image array (H, W, C)
        threshold: Threshold for white color detection (default: 240)

    Returns:
        Image with transparent background
    """
    # Convert numpy array to PIL Image with RGBA
    if image_array.shape[2] == 3:
        img = Image.fromarray(image_array).convert("RGBA")
    else:
        img = Image.fromarray(image_array)

    data = np.array(img).astype(np.float32)

    r, g, b, a = data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3]

    white_mask = (r > threshold) & (g > threshold) & (b > threshold)

    brightness = np.mean(data[:, :, :3], axis=2)
    alpha_new = 255 * (1 - np.clip((brightness - threshold) / (255 - threshold), 0, 1))
    data[:, :, 3] = np.minimum(a, alpha_new)

    data[:, :, 3][white_mask] = 0

    return Image.fromarray(data.astype(np.uint8))


def crop_to_nontransparent_area(img: Image.Image) -> Image.Image:
    """Crop transparent PNG to non-transparent bounding box.

    Args:
        img: PIL Image with transparency

    Returns:
        Cropped image
    """
    data = np.array(img)
    alpha = data[:, :, 3]

    nonzero = np.argwhere(alpha > 0)
    if nonzero.size == 0:
        return img

    y_min, x_min = nonzero.min(axis=0)
    y_max, x_max = nonzero.max(axis=0)

    return img.crop((x_min, y_min, x_max + 1, y_max + 1))


def save_docs_image(
    client: viser.ClientHandle,
    filename: str,
    remove_background: bool = True,
    threshold: int = 240,
    height: int = 720,
    width: int = 1280,
) -> None:
    """Save documentation image with optional background removal.

    Args:
        client: Viser client handle
        filename: Filename to save the image
        remove_background: Whether to remove white background (default: True)
        threshold: Threshold for white color detection (default: 240)
        height: Image height in pixels
        width: Image width in pixels
    """
    output_path = f"{DOC_IMAGES_DIR}/{filename}"

    # Get camera view as array
    image_array = get_camera_view_image(client, height=height, width=width)

    if remove_background:
        print(f"🔍 Removing background from {filename}...")
        img_no_bg = remove_white_background_from_array(image_array, threshold=threshold)
        cropped_img = crop_to_nontransparent_area(img_no_bg)
        cropped_img.save(output_path)
        print(f"✅ Saved: {filename}")
    else:
        # Save without background removal
        iio.imwrite(output_path, image_array, extension=".png")
        print(f"✅ Saved: {filename}")
