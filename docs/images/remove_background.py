import os
from pathlib import Path

import numpy as np
from PIL import Image


def remove_white_background(image_path: Path, threshold: int = 240) -> Image.Image:
    img = Image.open(image_path).convert("RGBA")
    data = np.array(img).astype(np.float32)

    r, g, b, a = data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3]

    white_mask = (r > threshold) & (g > threshold) & (b > threshold)

    brightness = np.mean(data[:, :, :3], axis=2)
    alpha_new = 255 * (1 - np.clip((brightness - threshold) / (255 - threshold), 0, 1))
    data[:, :, 3] = np.minimum(a, alpha_new)

    data[:, :, 3][white_mask] = 0

    return Image.fromarray(data.astype(np.uint8))


def crop_to_nontransparent_area(img: Image.Image) -> Image.Image:
    """Crop transparent PNG to non-transparent bounding box."""
    data = np.array(img)
    alpha = data[:, :, 3]

    nonzero = np.argwhere(alpha > 0)
    if nonzero.size == 0:
        return img

    y_min, x_min = nonzero.min(axis=0)
    y_max, x_max = nonzero.max(axis=0)

    return img.crop((x_min, y_min, x_max + 1, y_max + 1))


def process_current_script_dir() -> None:
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))

    for img_path in script_dir.glob("*.png"):
        print(f"🔍 Processing {img_path.name}...")
        img_no_bg = remove_white_background(img_path)
        cropped_img = crop_to_nontransparent_area(img_no_bg)

        out_path = img_path.with_name(img_path.stem + ".png")
        cropped_img.save(out_path)
        print(f"✅ Saved: {out_path.name}")


if __name__ == "__main__":
    process_current_script_dir()
