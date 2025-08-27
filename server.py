# server.py
import os, json, uuid, shutil, threading, time
from pathlib import Path
from typing import Dict, Any
from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from Codes.input_handler import tile_tiff, tile_numpy_image, write_tile_map
from Codes.image_enhancer import process_input_images
from Codes.detector import run_detection

import cv2

BASE = Path(__file__).parent.resolve()
PUBLIC = BASE / "public"
JOBS = BASE / "jobs"
OUTPUT_ROOT = BASE / "output_data" / "results_obb"
MODEL_PATH = BASE / "models" / "yolo11-obb.pt"

app = FastAPI()
# app.mount("/", StaticFiles(directory=str(PUBLIC), html=True), name="static")

# In-memory progress (also mirrored to disk per job for resilience)
PROGRESS: Dict[str, Dict[str, Any]] = {}
# --- helpers for deletion ---
def _safe_rmtree(p: Path):
    try:
        shutil.rmtree(p, ignore_errors=True)
    except Exception as e:
        print(f"[cleanup] warn: couldn't delete {p}: {e}")

def delete_job(job_id: str):
    """Remove all artifacts for a single job."""
    PROGRESS.pop(job_id, None)
    _safe_rmtree(JOBS / job_id)                 # working dir: orig/input/enh/meta/outputs
    _safe_rmtree(OUTPUT_ROOT / job_id)          # detector visualization (images/crops/csv)

def delete_all_history():
    """Remove everything (all jobs & outputs)."""
    PROGRESS.clear()
    _safe_rmtree(JOBS)
    _safe_rmtree(OUTPUT_ROOT)
    JOBS.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# --- Metadata helpers ---
def extract_raster_metadata(path: Path) -> dict:
    meta = {
        "filename": path.name,
        "size": path.stat().st_size,
        "is_geotiff": False,
        "has_georef": False,
    }
    try:
        import rasterio  # uses your existing dep
        with rasterio.open(path) as src:
            meta.update({
                "driver": src.driver,
                "width": src.width,
                "height": src.height,
                "count": src.count,
                "dtype": (src.dtypes[0] if src.count else None),
                "crs": (src.crs.to_string() if src.crs else None),
                "transform": tuple(src.transform) if src.transform else None,
                "bounds": [src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top],
                "res": list(src.res) if src.res else None,
            })
            meta["is_geotiff"] = (src.driver == "GTiff")
            meta["has_georef"] = (src.crs is not None)
            try:
                tags = src.tags()
                if "TIFFTAG_DATETIME" in tags:
                    meta["datetime"] = tags["TIFFTAG_DATETIME"]
            except Exception:
                pass
    except Exception:
        # Fallback for PNG/JPG
        im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if im is not None:
            h, w = im.shape[:2]
            meta.update({"driver": path.suffix.lower().lstrip("."), "width": w, "height": h})
    return meta

@app.get("/api/file/metadata")
async def file_metadata(fileId: str):
    p = ensure_dirs(fileId)
    f = p["meta"] / "metadata.json"
    if f.exists():
        return json.loads(f.read_text())
    return JSONResponse({"error": "not found"}, status_code=404)

# --- Space-saving cleanup ---
def nuke_all_previous():
    """Delete ALL previous jobs and result folders (destructive)."""
    PROGRESS.clear()
    if JOBS.exists():
        for p in JOBS.iterdir():
            shutil.rmtree(p, ignore_errors=True)
    if OUTPUT_ROOT.exists():
        for p in OUTPUT_ROOT.iterdir():
            shutil.rmtree(p, ignore_errors=True)

def cleanup_intermediates(job_id: str, keep_outputs: bool = True):
    """After finishing the run, delete heavy intermediates to save space."""
    # Delete per-tile annotated images/crops CSV stored under output_data/results_obb/<job_id>
    shutil.rmtree(OUTPUT_ROOT / job_id, ignore_errors=True)

    # Delete job working dirs (tiles, enhanced, meta). Keep outputs/ for downloads.
    paths = ensure_dirs(job_id)
    for k in ("input", "enh", "meta"):
        shutil.rmtree(paths[k], ignore_errors=True)

    # (Optional) if you also want to remove CSVs, move/copy what you need to outputs/ first.
def _read_bytes(p: Path) -> bytes:
    with open(p, "rb") as f:
        return f.read()

def ensure_dirs(job_id: str) -> Dict[str, Path]:
    job_dir = JOBS / job_id
    paths = {
        "job": job_dir,
        "orig": job_dir / "original",
        "input": job_dir / "input",        # tiles
        "enh": job_dir / "temp_upload",    # enhanced tiles
        "meta": job_dir / "meta",          # tile_map.json, progress.json
        "out": job_dir / "outputs",        # geojson + overlay
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def save_progress(job_id: str, data: Dict[str, Any]):
    PROGRESS[job_id] = data
    meta = ensure_dirs(job_id)["meta"]
    with open(meta / "progress.json", "w") as f:
        json.dump(data, f)


def load_progress(job_id: str) -> Dict[str, Any]:
    if job_id in PROGRESS:
        return PROGRESS[job_id]
    meta = ensure_dirs(job_id)["meta"]
    f = meta / "progress.json"
    if f.exists():
        return json.loads(f.read_text())
    return {"stage": "init", "percent": 0}


def set_stage(job_id: str, stage: str, **kw):
    p = load_progress(job_id)
    p.update({"stage": stage, **kw})
    save_progress(job_id, p)


# ---------- Upload API ----------

@app.post("/api/upload/init")
async def upload_init(req: Request):
    body = await req.json()
    filename = body.get("filename", "upload")
    job_id = uuid.uuid4().hex[:12]
    paths = ensure_dirs(job_id)
    (paths["orig"] / filename).with_suffix('')  # ensure folder exists
    save_progress(job_id, {"stage": "init", "filename": filename, "size": body.get("size", 0), "percent": 0})
    return {"fileId": job_id, "chunkSize": 5*1024*1024}


@app.post("/api/upload/chunk")
async def upload_chunk(request: Request):
    headers = request.headers
    job_id = headers.get("x-file-id")
    idx = int(headers.get("x-chunk-index", "0"))
    total = int(headers.get("x-chunk-total", "1"))
    if not job_id:
        return JSONResponse({"error": "missing file id"}, status_code=400)

    paths = ensure_dirs(job_id)
    tmp = paths["job"] / "upload.tmp"
    chunk = await request.body()
    with open(tmp, "ab") as f:
        f.write(chunk)

    set_stage(job_id, "uploading", message=f"Chunk {idx+1}/{total}")
    return {"ok": True}


@app.post("/api/upload/complete")
async def upload_complete(req: Request):
    body = await req.json()
    job_id = body["fileId"]
    p = load_progress(job_id)
    filename = p.get("filename", f"{job_id}.bin")
    paths = ensure_dirs(job_id)
    tmp = paths["job"] / "upload.tmp"
    final = paths["orig"] / filename
    shutil.move(tmp, final)
    meta_dict = extract_raster_metadata(final)
    (paths["meta"] / "metadata.json").write_text(json.dumps(meta_dict))
    set_stage(job_id, "uploaded", message="Upload complete", metadata=meta_dict, is_geotiff=bool(meta_dict.get("is_geotiff")))
    return {"ok": True}



# ---------- Model Info ----------

@app.get("/api/model/info")
async def model_info():
    try:
        # lazy import to keep startup light
        from ultralytics import YOLO
        model = YOLO(str(MODEL_PATH))
        classes = list(model.names.values())
        return {"model": MODEL_PATH.name, "classes": classes}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ---------- Processing ----------

def process_job(job_id: str, classes: list[str], params: Dict[str, Any]):
    paths = ensure_dirs(job_id)
    # Clean working dirs
    for k in ("input", "enh"):
        if paths[k].exists():
            shutil.rmtree(paths[k])
        paths[k].mkdir(parents=True, exist_ok=True)

    # Determine original file
    orig_files = list(paths["orig"].iterdir())
    if not orig_files:
        set_stage(job_id, "error", message="No original file")
        return
    orig = orig_files[0]

    # 1) Tiling (or assemble beforehand if you have chunk tiles pattern)
    set_stage(job_id, "tiling", percent=0, totalTiles=0, processedTiles=0)
    tile_size = int(params.get("tile_size", 1024))
    overlap = int(params.get("overlap", 256))

    if orig.suffix.lower() in (".tif", ".tiff"):
        total_tiles = tile_tiff(orig, name=job_id, tile_size=tile_size, overlap=overlap, use_rgb=False, out_dir=paths["input"], map_dir=paths["meta"])  # returns count
    else:
        img = cv2.imread(str(orig), cv2.IMREAD_UNCHANGED)
        total_tiles = tile_numpy_image(img, name=job_id, tile_size=tile_size, overlap=overlap, use_rgb=False, out_dir=paths["input"], map_dir=paths["meta"])  # returns count

    set_stage(job_id, "tiling", percent=5, totalTiles=total_tiles, processedTiles=0, message=f"{total_tiles} tiles queued")

    # 2) Enhance
    set_stage(job_id, "enhancing", percent=10)
    process_input_images(input_dir=paths["input"], output_dir=paths["enh"])  #  input/ → enh/

    # 3) Detect (updates progress per tile via callback)
    set_stage(job_id, "detecting", percent=20)

    def on_tile(i, n, last_tile_png=None):
        percent = 20 + int(70*(i/n)) if n else 20
        meta = {"stage": "detecting", "percent": percent, "totalTiles": n, "processedTiles": i}
        if last_tile_png:
            meta["lastTilePng"] = f"/api/preview/tile.png?fileId={job_id}&i={i}"
        set_stage(job_id, **meta)

    # Load metadata saved at upload_complete
    md_path = paths["meta"] / "metadata.json"
    meta = json.loads(md_path.read_text()) if md_path.exists() else {}
    is_geotiff = bool(meta.get("is_geotiff"))
    geojson_path = (paths["out"] / "detections.geojson") if is_geotiff else None
    orig_image_arg = str(orig) if is_geotiff else None

    run_detection(
        model_path=str(MODEL_PATH),
        input_dir=str(paths["enh"]),
        result_root=str(OUTPUT_ROOT),
        jobname=job_id,
        class_filter=set(c.lower() for c in classes) if classes else None,
        conf=float(params.get("conf", 0.20)),
        max_det=int(params.get("max_det", 600)),
        iou=float(params.get("iou", 0.25)),
        tile_map_path=str(paths["meta"] / "tile_map.json"),
        overlay_out=str(paths["out"] / "overlay.png"),
        geojson_out=(str(geojson_path) if geojson_path else None), 
        orig_image_path=orig_image_arg,                               
        progress_cb=on_tile,
    )

    overlay_ok = (paths["out"] / "overlay.png").exists()
    geojson_ok = bool(geojson_path and Path(geojson_path).exists())

    overlay_file = paths["out"] / "overlay.png"
    overlay_size = overlay_file.stat().st_size if overlay_ok else 0
    overlay_w = overlay_h = None
    if overlay_ok:
        try:
            img = cv2.imread(str(overlay_file), cv2.IMREAD_COLOR)
            overlay_h, overlay_w = img.shape[:2]
        except Exception:
            pass

    # Decide if we should recommend quadrants (very large = either huge pixels or > ~200MB)
    very_large = overlay_ok and (
        overlay_size > 200 * 1024 * 1024 or  # ~200MB
        (overlay_w and overlay_h and (overlay_w >= 15000 or overlay_h >= 15000))
    )
    quadrants = [f"/api/download/overlay_part?fileId={job_id}&q={i}" for i in range(1,5)] if very_large else None

    set_stage(
        job_id, "done", percent=100, message="Done",
        overlayReady=overlay_ok,
        overlayUrl=f"/api/download/overlay.png?fileId={job_id}" if overlay_ok else None,
        overlaySize=overlay_size,
        overlayWidth=overlay_w,
        overlayHeight=overlay_h,
        overlayQuadrants=quadrants,
        geojsonReady=geojson_ok,
        geojsonUrl=f"/api/download/geojson?fileId={job_id}" if geojson_ok else None,
    )

    try:
        cleanup_intermediates(job_id)
    except Exception as e:
        print(f"[cleanup] warning: {e}")



@app.post("/api/process/start")
async def process_start(req: Request):
    body = await req.json()
    job_id = body["fileId"]
    classes = body.get("classes", [])
    params = body.get("params", {})

    set_stage(job_id, "queued", percent=0, message="Starting")
    t = threading.Thread(target=process_job, args=(job_id, classes, params), daemon=True)
    t.start()
    return {"ok": True}


@app.get("/api/process/progress")
async def process_progress(fileId: str):
    return load_progress(fileId)


@app.post("/api/process/cancel")
async def process_cancel(fileId: str):
    # Simple soft-cancel: mark stage and let your worker honor a flag if you add one
    set_stage(fileId, "error", message="Cancelled")
    return {"ok": True}

from starlette.responses import Response as StarletteResponse  # if you want short alias

@app.post("/api/jobs/clear_current")
async def jobs_clear_current(request: Request, fileId: str | None = None):
    """
    Clears only the specified job's data (jobs/<id> + output_data/results_obb/<id>).
    If fileId not in query, tries JSON body: {"fileId": "..."}.
    """
    if not fileId:
        try:
            body = await request.json()
            fileId = body.get("fileId")
        except Exception:
            fileId = None
    if not fileId:
        return JSONResponse({"error": "fileId required"}, status_code=400)

    # soft-cancel so UI/worker don't keep writing while we delete
    try:
        set_stage(fileId, "error", message="Cleared by user")
    except Exception:
        pass

    # give any in-flight writes a moment to finish (optional)
    try:
        time.sleep(0.2)
    except Exception:
        pass

    delete_job(fileId)
    return JSONResponse({"ok": True})


@app.post("/api/jobs/clear_all")
async def jobs_clear_all():
    """
    Clears all jobs history (jobs/*) and all per-job output folders (output_data/results_obb/*).
    """
    delete_all_history()
    return JSONResponse({"ok": True})

# ---------- Downloads & Preview ----------


@app.get("/api/download/geojson")
async def download_geojson(fileId: str):
    f = ensure_dirs(fileId)["out"] / "detections.geojson"
    if not f.exists():
        return JSONResponse({"error": "not ready"}, status_code=404)
    return FileResponse(
        path=str(f),
        media_type="application/geo+json",
        filename=f"{fileId}.geojson",
        headers={"Cache-Control": "no-store"},
    )



@app.get("/api/preview/overlay.png")
async def preview_overlay(fileId: str):
    p = ensure_dirs(fileId)
    f = p["out"] / "overlay.png"
    if not f.exists():
        return JSONResponse({"error": "not ready"}, status_code=404)
    return FileResponse(str(f))

@app.get("/api/download/overlay_part")
async def download_overlay_part(fileId: str, q: int):
    """
    q in {1,2,3,4} mapping:
    1: top-left, 2: top-right, 3: bottom-left, 4: bottom-right
    """
    f = ensure_dirs(fileId)["out"] / "overlay.png"
    if not f.exists():
        return JSONResponse({"error": "not ready"}, status_code=404)

    img = cv2.imread(str(f), cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "bad image"}, status_code=500)

    H, W = img.shape[:2]
    midx, midy = W // 2, H // 2

    if q == 1:
        crop = img[0:midy, 0:midx]
    elif q == 2:
        crop = img[0:midy, midx:W]
    elif q == 3:
        crop = img[midy:H, 0:midx]
    elif q == 4:
        crop = img[midy:H, midx:W]
    else:
        return JSONResponse({"error": "q must be 1..4"}, status_code=400)

    ok, buf = cv2.imencode(".png", crop)
    if not ok:
        return JSONResponse({"error": "encode fail"}, status_code=500)

    return Response(content=buf.tobytes(), media_type="image/png", headers={"Cache-Control": "no-store"})

@app.get("/api/download/overlay.png")
async def download_overlay_png(fileId: str):
    f = ensure_dirs(fileId)["out"] / "overlay.png"
    if not f.exists():
        return JSONResponse({"error": "not ready"}, status_code=404)
    return FileResponse(
        path=str(f),
        media_type="image/png",
        filename=f"{fileId}_overlay.png",
        headers={"Cache-Control": "no-store"},
    )

@app.get("/api/preview/tile.png")
async def preview_tile(fileId: str, i: int = 0):
    res_dir = OUTPUT_ROOT / fileId / "images"
    if not res_dir.exists():
        return JSONResponse({"error": "not ready"}, status_code=404)
    files = sorted(res_dir.glob("*.png")) + sorted(res_dir.glob("*.jpg"))
    if not files:
        return JSONResponse({"error": "no tiles"}, status_code=404)
    i = max(0, min(i, len(files)-1))
    data = _read_bytes(files[i])  # read AFTER write finished
    return Response(content=data, media_type="image/png", headers={"Cache-Control": "no-store"})

def clear_job(job_id: str):
    # remove progress
    PROGRESS.pop(job_id, None)
    # remove job working dir
    shutil.rmtree(ensure_dirs(job_id)["job"], ignore_errors=True)
    # remove model result dir (tiles/annotated)
    shutil.rmtree(OUTPUT_ROOT / job_id, ignore_errors=True)

@app.post("/api/jobs/clear")
async def jobs_clear(fileId: str):
    clear_job(fileId)
    return {"ok": True}


# Root index fallback
@app.get("/app")
async def app_index():
    index = PUBLIC / "index.html"
    return HTMLResponse(index.read_text())



# Serve UI at /
@app.get("/", include_in_schema=False)
async def root_index():
    index = PUBLIC / "index.html"
    return FileResponse(str(index))

# Optional: favicon to avoid 404 spam
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    ico = PUBLIC / "favicon.ico"
    if ico.exists():
        return FileResponse(str(ico))
    return Response(status_code=204)

# Mount static files (css/js/assets) somewhere safe (not "/")

app.mount("/public", StaticFiles(directory=str(PUBLIC), html=False), name="public")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)