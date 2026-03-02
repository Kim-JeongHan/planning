import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
PROJECT_CONFIG_DIR = PROJECT_ROOT / "config"
DOC_DIR = pathlib.Path(__file__).parent.parent / "docs"
DOC_IMAGES_DIR = DOC_DIR / "images"
