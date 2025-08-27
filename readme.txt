project/
├─ server.py                      # FastAPI backend (chunk upload + pipeline + downloads)                # CLI orchestrator (optional, same pipeline without web)
├─ requirements.txt
├─ models/
│  └─ yolo11-obb.pt              # Put your model file here
├─ output_data/                   # results go here (per image/job subfolders)
├─ jobs/                          # per-upload working dirs (created at runtime)
├─ public/
│  └─ index.html                  # Web UI (already built)
└─ Codes/
   ├─ input_handler.py            # tiling helpers (non-interactive) + map json writing
   ├─ image_enhancer.py           # enhancement util (parametrized I/O dirs)
   ├─ detector.py                 # detection, aggregation, GeoJSON + overlay
   ├─ postprocess.py              # stitch overlay, build geojson
   └─ chunk_assembler.py          # optional: stitch chunked tiles to one image