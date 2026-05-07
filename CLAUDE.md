# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

LookLabeledData is a local browser-based tool for visualizing training datasets. It reads PyTorch `.pt` files and COCO JSON datasets, renders images with annotation overlays (bbox/polygon/keypoint) in a browser via HTML5 Canvas.

## Start / develop

```bash
python3 app.py --port 8787   # starts FastAPI + uvicorn, auto-opens browser
```

Backend-only dev (no browser auto-open):
```bash
uvicorn app:app --host 127.0.0.1 --port 8787 --reload
```

Dependencies: `fastapi`, `uvicorn`, `torch`, `pillow`, `opencv-python-headless`.

## Architecture

Single-file backend (`app.py`) + inline JS/CSS frontend (`static/index.html`, `static/style.css`).

- **Backend**: FastAPI with routes under `/api/` (`/api/browse`, `/api/pt/detect`, `/api/pt/load`, `/api/coco/info`, `/api/coco/load`). Temp images served via `/api/temp/{filename}`.
- **Frontend**: Plain ES5 JS inlined in `index.html` (self-contained, no framework, no build step). All DOM references cached in `E` object. State in `state` object.

## Key patterns

### Tensor ↔ numpy handling

The codebase deals with both `torch.Tensor` and `numpy.ndarray` in `.pt` files. Use `_to_numpy(t)` helper (defined in `app.py`) **everywhere** before calling `.detach()` — it safely handles both types. Never call `.detach()` directly.

### Temp file lifecycle

`/tmp/looklabeled_<pid>/` stores converted PNGs. Cleaned up on:
- Normal exit (`atexit`)
- SIGINT/SIGTERM (signal handler)
- Startup (cleans orphans from dead processes)

### API contract for samples

`pt_load` and `coco_load` both return `{"samples": [...]}`. Each sample has:
```json
{
  "filename": "data_xxx.pt",
  "image_url": "/api/temp/abc123.png",
  "width": 512, "height": 512,
  "annotations": [{"bbox": [x1,y1,x2,y2], "polygons": [...], "label": 3}]
}
```

### COCO bbox format

COCO uses `[x, y, width, height]` — always converted to `[x1, y1, x2, y2]` before sending to frontend.

### Polygon format (dual representation)

Canvas renderer in `drawOverlay()` handles two polygon formats:
- Array of points: `[[x1,y1], [x2,y2], ...]`
- Flat array: `[x1, y1, x2, y2, ...]` (COCO standard)

### Fontend image sizing

Image and canvas are explicitly sized to `wrapper.clientWidth * 0.92 / clientHeight * 0.75` aspect-ratio-fitted dimensions. Canvas is a sibling overlay (absolute positioned) on the image wrapper. Do NOT use `object-fit: contain` on the image — it breaks canvas alignment.

## Common debugging

- Frontend JS errors appear in browser Console (Cmd+Option+J on macOS Chrome)
- Backend errors appear in terminal; all unhandled exceptions print full tracebacks via global exception handler
- To inspect `.pt` file structure, use `/api/pt/detect` endpoint
- Stale `__pycache__/` can cause old code to run — delete it if behavior doesn't match source
