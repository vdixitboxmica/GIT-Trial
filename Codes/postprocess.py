# Codes/postprocess.py
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import cv2
import numpy as np

# ---- Tile map helpers ----
def load_tile_map(tile_map_path: Path) -> Tuple[int, int, List[Dict[str, Any]]]:
    data = json.loads(Path(tile_map_path).read_text())
    canvas = data.get("canvas", {})
    tiles = data.get("tiles", [])
    W = int(canvas.get("width", 0))
    H = int(canvas.get("height", 0))
    return W, H, tiles

def _class_color(name: str) -> Tuple[int, int, int]:
    # BGR colors (matches your palette reasonably)
    base = {
        "small-vehicle": (255, 0, 0),     # red
        "large-vehicle": (0, 255, 255),   # yellow
        "ship": (128, 0, 255),            # purple
    }
    return base.get(name, (0, 255, 0))    # default = green

# ---- Robust overlay builder (does NOT depend on annotated tiles) ----
def build_overlay_from_tiles_and_records(
    tile_map_path: Path,
    tiles_dir: Path,
    records: List[Dict[str, Any]],
    overlay_out: Path,
    draw_thickness: int = 2,
) -> None:
    """
    Rebuild the full image from the ENHANCED tiles and draw detections using
    pixel-global coordinates ('pts_global' in records). This avoids reading
    per-tile annotated images.
    """
    W, H, tiles = load_tile_map(tile_map_path)
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    tiles_dir = Path(tiles_dir)
    # Stitch base imagery from tiles
    for t in tiles:
        x, y, w, h = int(t["x"]), int(t["y"]), int(t["w"]), int(t["h"])
        fname = t["file"]
        src = cv2.imread(str(tiles_dir / fname), cv2.IMREAD_COLOR)
        if src is None:
            print(f"[overlay] warn: missing tile {tiles_dir / fname}")
            continue
        src = src[0:h, 0:w]
        canvas[y:y+h, x:x+w] = src

    # Draw detections (global pixel space)
    for r in records:
        pts = r.get("pts_global")
        if not pts:
            continue
        try:
            pts_arr = np.array(pts, dtype=int)
            color = _class_color(r.get("class", ""))
            cv2.polylines(canvas, [pts_arr], True, color, draw_thickness)
        except Exception as e:
            print(f"[overlay] draw error: {e}")

    overlay_out.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(overlay_out), canvas)
    if not ok:
        print(f"[overlay] warn: failed to write {overlay_out}")

# ---- Legacy stitcher (reads annotated tiles). Kept for compatibility. ----
def stitch_overlay(
    tile_map_path: Path,
    images_dir: Path,
    overlay_out: Path,
) -> None:
    """
    Legacy: stitches overlay by re-reading per-tile *annotated* images in images_dir.
    If your build saves annotated tiles with the same filenames as the original tiles,
    this works; otherwise prefer build_overlay_from_tiles_and_records().
    """
    W, H, tiles = load_tile_map(tile_map_path)
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    images_dir = Path(images_dir)

    for t in tiles:
        x, y, w, h = int(t["x"]), int(t["y"]), int(t["w"]), int(t["h"])
        fname = t["file"]
        src = cv2.imread(str(images_dir / fname), cv2.IMREAD_COLOR)
        if src is None:
            print(f"[stitch_overlay] warn: missing annotated {images_dir / fname}")
            continue
        src = src[0:h, 0:w]
        canvas[y:y+h, x:x+w] = src

    overlay_out.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(overlay_out), canvas)
    if not ok:
        print(f"[stitch_overlay] warn: failed to write {overlay_out}")

# ---- GeoJSON writer (pixel-space by default) ----
# Optional: only needed for georeferencing when a GeoTIFF is provided
try:
    import rasterio
except Exception:
    rasterio = None  # will gracefully fall back to pixel coords

def detections_to_geojson(
    records: List[Dict[str, Any]],
    tile_map_path: Optional[Path],
    out_path: Path,
    orig_image_path: Optional[Path] = None,   # <-- NEW optional 4th param
) -> None:
    """
    Write a GeoJSON FeatureCollection of OBB polygons.

    - If `orig_image_path` is a GeoTIFF and rasterio is available, polygons are
      emitted in the source CRS, using the GeoTIFF's affine transform.
    - Otherwise, polygons are emitted in pixel coordinates (and a note is added).
    """
    crs_str = None
    to_world = None

    # Build pixel->world transformer if possible
    if orig_image_path and rasterio is not None:
        try:
            with rasterio.open(orig_image_path) as src:
                transform = src.transform
                crs = src.crs
                crs_str = crs.to_string() if crs else None

                def _to_world(pt_xy):
                    x, y = pt_xy
                    X, Y = transform * (x, y)  # affine transform
                    return [float(X), float(Y)]

                to_world = _to_world
        except Exception as e:
            print(f"[geojson] warning: georeferencing unavailable ({e}); falling back to pixel coords.")

    features: List[Dict[str, Any]] = []

    for rec in records:
        pts = rec.get("pts_global")
        if not pts or len(pts) != 4:
            # Skip malformed records; detector always supplies 4 points
            continue

        # Build coordinates (close the ring)
        if to_world:
            ring = [to_world(p) for p in pts] + [to_world(pts[0])]
        else:
            ring = [[float(p[0]), float(p[1])] for p in pts]
            ring.append(ring[0][:])

        props = {
            "tile": rec.get("tile"),
            "class": rec.get("class"),
            "confidence": float(rec.get("confidence", 0.0)),
            "width_px": float(rec.get("width", 0.0)),
            "height_px": float(rec.get("height", 0.0)),
            "angle_rad": float(rec.get("angle_rad", 0.0)),
        }

        feat = {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": props,
        }
        features.append(feat)

    fc: Dict[str, Any] = {"type": "FeatureCollection", "features": features, "properties": {}}
    if crs_str:
        # Note: RFC 7946 discourages non-WGS84 coords, but many GIS apps accept this metadata.
        fc["properties"]["crs"] = crs_str
    elif to_world is None:
        fc["properties"]["note"] = "Pixel coordinates; source image not georeferenced."

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(fc))