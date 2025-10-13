import os
from pathlib import Path

import numpy as np
from PIL import Image


def crop_to_nontranspare1nt_area(image_path: Path, output_path: Path) -> None:
    """Crop transparent PNG to non-transparent bounding box."""
    img = Image.open(image_path).convert("RGBA")
    data = np.array(img)
    alpha = data[:, :, 3]

    # 투명하지 않은 픽셀 영역만 추출
    nonzero = np.argwhere(alpha > 0)
    if nonzero.size == 0:
        print(f"⚠️ {image_path.name}: no visible region, skipped.")
        return

    y_min, x_min = nonzero.min(axis=0)
    y_max, x_max = nonzero.max(axis=0)

    cropped = img.crop((x_min, y_min, x_max + 1, y_max + 1))
    cropped.save(output_path)
    print(f"✅ Cropped: {output_path.name}")


def process_current_script_dir() -> None:
    # 현재 실행 중인 Python 파일 경로
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))

    for img_path in script_dir.glob("*.png"):
        out_path = img_path.with_name(img_path.stem + ".png")
        crop_to_nontranspare1nt_area(img_path, out_path)


if __name__ == "__main__":
    process_current_script_dir()
