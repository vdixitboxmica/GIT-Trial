# Codes/detector.py
from ultralytics import YOLO
import cv2, pandas as pd
import numpy as np
from matplotlib import cm
from pathlib import Path
import json, time
from typing import Callable, Optional, Set

from .postprocess import detections_to_geojson, build_overlay_from_tiles_and_records

manual_color_map = {
    "small-vehicle": (255, 0, 0),
    "large-vehicle": (0, 255, 255),
    "ship": (128, 0, 255),
}

def run_detection(
    model_path: str,
    input_dir: str,
    result_root: str,
    jobname: str,
    class_filter: Optional[Set[str]] = None,
    color_mode: str = "manual",
    conf: float = 0.20,          # safer default
    max_det: int = 600,          # safer default
    iou: float = 0.25,
    tile_map_path: Optional[str] = None,
    overlay_out: Optional[str] = None,
    geojson_out: Optional[str] = None,
    progress_cb: Optional[Callable[[int, int, Optional[str]], None]] = None,
    orig_image_path: Optional[str] = None,   # <-- add this
):
    def get_class_color(name, idx=None):
        if color_mode == "manual":
            return manual_color_map.get(name, (255, 255, 255))
        cmap = cm.get_cmap("hsv", 15)
        rgb = np.array(cmap(idx)[:3]) * 255
        return tuple(int(x) for x in rgb[::-1])  # RGB->BGR

    input_dir = Path(input_dir)
    result_root = Path(result_root) / jobname
    img_save_dir = result_root / "images"
    crop_dir = result_root / "crops"
    csv_output_path = result_root / "detections.csv"
    img_save_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path)
    names = model.names

    image_files = [p for p in sorted(input_dir.iterdir()) if p.suffix.lower() in {".jpg", ".png", ".tif", ".jpeg"}]
    N = len(image_files)
    records = []

    # Load tile offsets for global coords
    offsets = {}
    if tile_map_path and Path(tile_map_path).exists():
        tile_map = json.loads(Path(tile_map_path).read_text())
        offsets = {Path(t["file"]).stem: (t["x"], t["y"]) for t in tile_map.get("tiles", [])}

    for i, image_path in enumerate(image_files, start=1):
        # Eager tick so UI shows movement even during long predicts
        if progress_cb:
            progress_cb(i - 1, N)

        tile_img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if tile_img is None:
            if progress_cb:
                progress_cb(i, N)
            continue

        annotated = tile_img.copy()

        # Timed predict with sane caps
        try:
            t0 = time.time()
            res = model.predict(
                source=tile_img,
                save=False,
                conf=conf,
                max_det=max_det,
                iou=iou,
                verbose=False,
            )[0]
            dt = time.time() - t0
            if dt > 15:
                print(f"[{image_path.name}] WARNING: slow tile took {dt:.1f}s (consider lowering max_det or raising conf)")
        except Exception as e:
            print(f"[{image_path.name}] predict ERROR: {e}")
            cv2.imwrite(str(img_save_dir / image_path.name), annotated)  # save unannotated so preview can advance
            if progress_cb:
                progress_cb(i, N, str(img_save_dir / image_path.name))
            continue

        # Guard: no OBBs
        if not hasattr(res, "obb") or res.obb is None or len(res.obb) == 0:
            cv2.imwrite(str(img_save_dir / image_path.name), annotated)
            if progress_cb:
                progress_cb(i, N, str(img_save_dir / image_path.name))
            continue

        off = offsets.get(image_path.stem, (0, 0))

        for box in res.obb:
            obb = box.data.tolist()[0]
            if len(obb) < 7:
                continue
            x, y, w, h, angle, score, cls_id = obb
            name = names[int(cls_id)]
            if class_filter and (name.lower() not in class_filter):
                continue

            rect = ((x, y), (w, h), angle * 180.0 / np.pi)
            pts = cv2.boxPoints(rect).astype(int)
            if cv2.contourArea(pts) <= 0.5:
                continue

            # Draw on tile
            color = get_class_color(name, int(cls_id))
            cv2.polylines(annotated, [pts], True, color, 2)
            cv2.putText(annotated, name, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Safe crop (optional)
            mask = np.zeros_like(tile_img)
            cv2.drawContours(mask, [pts], 0, (255, 255, 255), -1)
            masked = cv2.bitwise_and(tile_img, mask)
            x0, y0, w0, h0 = cv2.boundingRect(pts)
            H, W = tile_img.shape[:2]
            x0 = max(0, x0); y0 = max(0, y0)
            x1 = min(x0 + w0, W); y1 = min(y0 + h0, H)
            if x1 - x0 >= 1 and y1 - y0 >= 1:
                crop = masked[y0:y1, x0:x1]
                if crop is not None and crop.size > 0:
                    crop_name = f"{image_path.stem}_det_{len(records)}.jpg"
                    cv2.imwrite(str(crop_dir / crop_name), crop)

            # One record per detection
            pts_global = (pts + np.array(off)).tolist()
            records.append({
                "tile": image_path.name,
                "class": name,
                "confidence": float(score),
                "x_center": float(x + off[0]),
                "y_center": float(y + off[1]),
                "width": float(w),
                "height": float(h),
                "angle_rad": float(angle),
                "pts_global": pts_global,
            })

        # Save annotated tile & advance progress
        cv2.imwrite(str(img_save_dir / image_path.name), annotated)
        if progress_cb:
            progress_cb(i, N, str(img_save_dir / image_path.name))

    # After all tiles: persist CSV
    if records:
        pd.DataFrame(records).to_csv(csv_output_path, index=False)

    # Aggregate outputs (overlay + geojson)
    if tile_map_path and overlay_out:
        build_overlay_from_tiles_and_records(
            Path(tile_map_path), Path(input_dir),  # enhanced tiles dir
            records,
            Path(overlay_out),
            draw_thickness=2,
        )
    if geojson_out:
        detections_to_geojson(
            records,
            Path(tile_map_path) if tile_map_path else None,
            Path(geojson_out),
            Path(orig_image_path) if orig_image_path else None,  # now defined
        )
