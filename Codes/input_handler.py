# Codes/input_handler.py
from pathlib import Path
import cv2, rasterio, numpy as np, json
from rasterio.windows import Window


def write_tile_map(map_path: Path, canvas_w: int, canvas_h: int, tiles: list[dict]):
    map_path.parent.mkdir(parents=True, exist_ok=True)
    data = {"canvas": {"width": canvas_w, "height": canvas_h}, "tiles": tiles}
    map_path.write_text(json.dumps(data))


def _save_tile(out_dir: Path, name: str, x: int, y: int, tile, tile_size: int):
    h, w = tile.shape[:2]
    pad_bottom = tile_size - h
    pad_right = tile_size - w
    if pad_bottom > 0 or pad_right > 0:
        tile = cv2.copyMakeBorder(tile, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=(0,0,0))
    out = out_dir / f"{name}_x{x}_y{y}.png"
    cv2.imwrite(str(out), tile)
    return out


def tile_numpy_image(image, name: str, tile_size=1024, overlap=256, use_rgb=False, out_dir: Path = Path("input"), map_dir: Path = Path("meta")) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    H, W = image.shape[:2]
    if image.shape[-1] > 3:
        image = image[:, :, :3]
    if use_rgb:
        image = image[:, :, ::-1]

    tiles_meta = []
    count = 0
    for y in range(0, H, tile_size - overlap):
        for x in range(0, W, tile_size - overlap):
            tile = image[y:min(y+tile_size, H), x:min(x+tile_size, W)]
            _save_tile(out_dir, name, x, y, tile, tile_size)
            tiles_meta.append({"x": x, "y": y, "w": tile.shape[1], "h": tile.shape[0], "file": f"{name}_x{x}_y{y}.png"})
            count += 1
    write_tile_map(map_dir / "tile_map.json", W, H, tiles_meta)
    return count


def tile_tiff(image_path: Path, name: str, tile_size=1024, overlap=256, use_rgb=False, out_dir: Path = Path("input"), map_dir: Path = Path("meta")) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(image_path) as src:
        W, H = src.width, src.height
        tiles_meta = []
        count = 0
        for y in range(0, H, tile_size - overlap):
            for x in range(0, W, tile_size - overlap):
                window = Window(x, y, min(tile_size, W-x), min(tile_size, H-y))
                tile = src.read(window=window)  # C,H,W
                tile = np.transpose(tile, (1,2,0))
                if tile.shape[2] > 3:
                    tile = tile[:, :, :3]
                tile = cv2.convertScaleAbs(tile)
                if use_rgb:
                    tile = tile[:, :, ::-1]
                _save_tile(out_dir, name, x, y, tile, tile_size)
                tiles_meta.append({"x": x, "y": y, "w": tile.shape[1], "h": tile.shape[0], "file": f"{name}_x{x}_y{y}.png"})
                count += 1
    write_tile_map(map_dir / "tile_map.json", W, H, tiles_meta)
    return count