import os
import sys
import json
import atexit
import signal
import shutil
import random
import tempfile
import time
import hashlib
import traceback
from pathlib import Path
from io import BytesIO
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# ---------------------------------------------------------------------------
# Optional imports – torch is required for .pt support
# ---------------------------------------------------------------------------
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# ============================================================================
# Temp file management
# ============================================================================
TEMP_ROOT = Path(tempfile.gettempdir()) / f"looklabeled_{os.getpid()}"


def _cleanup():
    if TEMP_ROOT.exists():
        shutil.rmtree(TEMP_ROOT, ignore_errors=True)


def _signal_handler(signum, frame):
    _cleanup()
    sys.exit(0)


# Clean up old temp dirs from previous runs (dead processes)
for _d in Path(tempfile.gettempdir()).glob("looklabeled_*"):
    try:
        shutil.rmtree(_d, ignore_errors=True)
    except Exception:
        pass

TEMP_ROOT.mkdir(parents=True, exist_ok=True)
atexit.register(_cleanup)
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

# ============================================================================
# FastAPI app
# ============================================================================
app = FastAPI(title="LookLabeledData")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {exc}"},
    )

# Mount static files AFTER defining routes, but we define routes first
# Static mount happens at the bottom to avoid route conflicts

# ============================================================================
# Pydantic models
# ============================================================================


class PTDetectRequest(BaseModel):
    path: str
    file: str


class PTLoadRequest(BaseModel):
    path: str
    files: list[str]
    image_key: str = ""
    bbox_key: str = ""
    label_key: str = ""
    mask_key: str = ""
    keypoint_key: str = ""
    bbox_mode: str = "xyxy"  # xyxy | xywh


class COCOLoadRequest(BaseModel):
    path: str
    count: int = 10


# ============================================================================
# COCO annotation cache
# ============================================================================
_coco_cache: dict[str, dict] = {}


# ============================================================================
# Helpers – image tensor conversion
# ============================================================================

def tensor_to_pil(t) -> Image.Image:
    """Convert a torch tensor or numpy array to PIL Image."""
    if isinstance(t, Image.Image):
        return t
    if isinstance(t, np.ndarray):
        arr = t
    elif hasattr(t, "detach"):
        arr = t.detach().cpu().numpy()
    else:
        arr = np.asarray(t)
    if arr.ndim == 3:
        if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = arr.transpose(1, 2, 0)
        elif arr.shape[0] == arr.shape[1] == arr.shape[2]:
            # Ambiguous 3xNxM – assume channels-first if small first dim
            if arr.shape[0] in (1, 3, 4):
                arr = arr.transpose(1, 2, 0)
    # Normalise float [0,1] or [-1,1] → uint8
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        if arr.min() < 0:
            arr = (arr + 1) / 2.0
        arr = (arr * 255).clip(0, 255).astype(np.uint8)

    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L")
    elif arr.shape[2] == 1:
        return Image.fromarray(arr[:, :, 0], mode="L")
    elif arr.shape[2] == 3:
        return Image.fromarray(arr, mode="RGB")
    elif arr.shape[2] == 4:
        return Image.fromarray(arr, mode="RGBA")
    return Image.fromarray(arr[:, :, :3], mode="RGB")


def save_tensor_as_png(t, filename: str) -> Path:
    """Save tensor as PNG in temp dir, return path."""
    img = tensor_to_pil(t)
    dest = TEMP_ROOT / filename
    img.save(dest, "PNG")
    return dest


def tensor_hash(t) -> str:
    """Short hash for tensor content to avoid duplicate temp files."""
    try:
        if isinstance(t, np.ndarray):
            data = t.tobytes()
        elif hasattr(t, "detach"):
            data = t.detach().cpu().numpy().tobytes()
        else:
            data = np.asarray(t).tobytes()
    except Exception:
        data = str(id(t)).encode()
    return hashlib.md5(data[:4096]).hexdigest()[:12]


# ============================================================================
# Helpers – PT key detection
# ============================================================================

def _suggest_role(name: str, shape: tuple, dtype_str: str) -> str:
    """Heuristically guess the role of a tensor key."""
    ndim = len(shape)

    # Image detection: 3-dim, last or first dim is channel (1/3/4)
    if ndim == 3:
        if shape[0] in (1, 3, 4) and shape[-1] not in (1, 3, 4):
            return "image"
        if shape[-1] in (1, 3, 4) and shape[0] not in (1, 3, 4):
            return "image"

    # BBox detection: [N, 4] or [N, 4, 2] or [N, 5]
    if ndim == 2:
        if shape[1] == 4:
            return "bbox"
        if shape[1] == 5:
            return "bbox_label"

    # Label detection: [N] with integer dtype
    if ndim == 1 and "int" in dtype_str:
        return "label"

    # Mask detection: [N, H, W] or [H, W]
    if ndim == 3 and shape[0] > 0:
        # Could be mask stack or temporal sequence
        if "int" in dtype_str or "bool" in dtype_str or "uint8" in dtype_str:
            return "mask"
    if ndim == 2 and ("int" in dtype_str or "bool" in dtype_str or "uint8" in dtype_str):
        return "mask"

    # Keypoint detection
    if ndim == 3 and shape[2] in (2, 3) and shape[1] > 0:
        return "keypoint"

    # Name-based heuristics
    nl = name.lower()
    if any(k in nl for k in ("image", "img", "picture", "photo")):
        return "image"
    if any(k in nl for k in ("bbox", "box", "boxes", "gt_box")):
        return "bbox"
    if any(k in nl for k in ("label", "class", "category", "cls", "target")):
        return "label"
    if any(k in nl for k in ("mask", "seg", "segment")):
        return "mask"
    if any(k in nl for k in ("keypoint", "kpt", "landmark", "joint")):
        return "keypoint"

    return "unknown"


# ============================================================================
# Helpers – Annotation conversion for PT
# ============================================================================

def _to_numpy(t):
    if isinstance(t, np.ndarray):
        return t
    return t.detach().cpu().numpy()

def _convert_bbox(bbox_tensor, bbox_mode: str, label_tensor=None):
    """Convert bbox tensor to list of dicts with xyxy format."""
    if bbox_tensor is None:
        return []
    arr = _to_numpy(bbox_tensor)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    annotations = []
    for i in range(arr.shape[0]):
        row = arr[i]
        if len(row) >= 4:
            if bbox_mode == "xywh":
                x1, y1, w, h = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                x2, y2 = x1 + w, y1 + h
            else:  # xyxy
                x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            ann = {"bbox": [x1, y1, x2, y2]}
            if label_tensor is not None and i < len(label_tensor):
                ann["label"] = int(label_tensor[i].item()) if hasattr(label_tensor[i], 'item') else int(label_tensor[i])
            elif len(row) == 5:
                ann["label"] = int(row[4])
            annotations.append(ann)
    return annotations


def _convert_mask(mask_tensor):
    """Convert mask tensor to overlay image URLs + polygon contours."""
    if mask_tensor is None:
        return []
    arr = _to_numpy(mask_tensor)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]

    results = []
    for i in range(min(arr.shape[0], 100)):  # cap at 100 masks
        m = arr[i]
        if m.dtype != np.uint8:
            m = (m > 0).astype(np.uint8) * 255

        polygons = []
        if HAS_CV2:
            try:
                contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    if len(cnt) >= 3:
                        pts = cnt.reshape(-1, 2).tolist()
                        polygons.append(pts)
            except Exception:
                pass

        results.append({
            "index": i,
            "polygons": polygons,
        })
    return results


def _convert_keypoint(kpt_tensor):
    """Convert keypoint tensor to list of point lists."""
    if kpt_tensor is None:
        return None
    arr = _to_numpy(kpt_tensor)
    return arr.tolist()


# ============================================================================
# Helpers – COCO
# ============================================================================

def _find_coco_files(root: Path) -> Optional[dict]:
    """Find COCO annotation JSON and image directory under root."""
    root = Path(root)
    if not root.exists():
        return None

    json_files = sorted(root.glob("*.json"))
    # Prefer annotation files: instances_*, annotations, coco_*
    anno_json = None
    for jf in json_files:
        name = jf.name.lower()
        if "instance" in name or "annotation" in name or "coco" in name:
            anno_json = jf
            break
    if anno_json is None and json_files:
        anno_json = json_files[0]

    if anno_json is None:
        return None

    # Find image directory
    image_dir = None
    for candidate in [
        root / "images",
        root / "train2017", root / "val2017",
        root / "train", root / "val",
    ]:
        if candidate.exists():
            image_dir = candidate
            break
    # Fallback: look for any subdirectory with images
    if image_dir is None:
        for sub in sorted(root.iterdir()):
            if sub.is_dir():
                imgs = list(sub.glob("*.jpg")) + list(sub.glob("*.png"))
                if imgs:
                    image_dir = sub
                    break

    return {"json": anno_json, "image_dir": image_dir}


def _load_coco_json(path: Path) -> dict:
    """Load COCO JSON with caching."""
    key = str(path)
    if key in _coco_cache:
        return _coco_cache[key]
    with open(path, "r") as f:
        data = json.load(f)
    _coco_cache[key] = data
    return data


def _decode_coco_rle(rle: dict) -> np.ndarray:
    """Decode COCO RLE (uncompressed, list counts) to binary mask."""
    counts = rle["counts"]
    h, w = rle["size"]
    if isinstance(counts, str):
        raise NotImplementedError("Base64 RLE requires pycocotools")
    mask = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    for i, count in enumerate(counts):
        if i % 2 == 1:
            mask[pos:pos + count] = 1
        pos += count
    return mask.reshape((h, w), order="F")


def _polygons_from_mask(mask: np.ndarray) -> list:
    """Extract polygons from binary mask using cv2 or simple edge walk."""
    if HAS_CV2:
        contours, _ = cv2.findContours(
            (mask * 255).astype(np.uint8),
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        return [cnt.reshape(-1, 2).tolist() for cnt in contours if len(cnt) >= 3]
    return []


# ============================================================================
# Helpers – Palette
# ============================================================================

def _color_for_id(cat_id: int) -> str:
    """Deterministic colour per category id."""
    palette = [
        "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
        "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabebe",
        "#469990", "#e6beff", "#9a6324", "#800000", "#aaffc3",
        "#808000", "#ffd8b1", "#000075", "#a9a9a9", "#000000",
    ]
    return palette[cat_id % len(palette)]


# ============================================================================
# API Routes
# ============================================================================

@app.get("/api/browse")
async def browse(path: str = Query("/")):
    """Browse a directory, returning subdirs and files."""
    def _empty_resp(error_msg):
        return JSONResponse({
            "error": error_msg,
            "path": path,
            "parent": str(Path(path).parent) if Path(path).parent != Path(path) else "/",
            "dirs": [], "files": [], "pt_files": [], "json_files": [],
        })

    try:
        p = Path(path)
        if not p.exists():
            return _empty_resp(f"Path does not exist: {path}")
        try:
            entries = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        except PermissionError:
            return _empty_resp(f"Permission denied: {path}")
    except Exception as e:
        import traceback; traceback.print_exc()
        return _empty_resp(f"Error accessing path: {e}")

    dirs = []
    files = []
    pt_files = []
    json_files = []

    for e in entries:
        if e.name.startswith("."):
            continue
        if e.is_dir():
            dirs.append({"name": e.name, "path": str(e)})
        elif e.is_file():
            files.append({"name": e.name, "path": str(e)})
            if e.suffix.lower() in (".pt", ".pth"):
                pt_files.append({"name": e.name, "path": str(e)})
            if e.suffix.lower() == ".json":
                json_files.append({"name": e.name, "path": str(e)})

    return JSONResponse({
        "error": None,
        "path": path,
        "parent": str(p.parent) if p.parent != p else "/",
        "dirs": dirs,
        "files": files,
        "pt_files": pt_files,
        "json_files": json_files,
    })


@app.post("/api/pt/detect")
async def pt_detect(req: PTDetectRequest):
    """Read one .pt file and return its keys with shape/types and role suggestions."""
    if not HAS_TORCH:
        raise HTTPException(500, "torch is not installed")

    full_path = Path(req.path) / req.file
    if not full_path.exists():
        raise HTTPException(404, f"File not found: {full_path}")

    try:
        data = torch.load(full_path, map_location="cpu", weights_only=False)
    except Exception as e:
        raise HTTPException(400, f"Failed to load .pt file: {e}")

    if not isinstance(data, dict):
        raise HTTPException(400, f"Expected a dict, got {type(data).__name__}")

    keys_info = []
    for key, value in data.items():
        info = {"name": key, "type": type(value).__name__}
        if isinstance(value, torch.Tensor):
            info["shape"] = list(value.shape)
            info["dtype"] = str(value.dtype)
            info["suggested_role"] = _suggest_role(key, tuple(value.shape), str(value.dtype))
        elif isinstance(value, np.ndarray):
            info["shape"] = list(value.shape)
            info["dtype"] = str(value.dtype)
            info["suggested_role"] = _suggest_role(key, tuple(value.shape), str(value.dtype))
        elif isinstance(value, list):
            info["len"] = len(value)
            if value and isinstance(value[0], torch.Tensor):
                info["element_shape"] = list(value[0].shape)
                info["element_dtype"] = str(value[0].dtype)
            info["suggested_role"] = _suggest_role(key, (len(value),), "list")
        elif isinstance(value, dict):
            info["keys"] = list(value.keys())
            info["suggested_role"] = "dict"
        else:
            info["suggested_role"] = "other"
        keys_info.append(info)

    return JSONResponse({"file": req.file, "keys": keys_info})


@app.post("/api/pt/load")
async def pt_load(req: PTLoadRequest):
    """Load .pt files, convert images to PNGs, return annotations."""
    if not HAS_TORCH:
        raise HTTPException(500, "torch is not installed")

    print(f"[pt_load] path={req.path}, files={len(req.files)}, image_key={req.image_key!r}")
    debug_printed = False
    image_key = req.image_key  # user-specified key, may be empty
    ok_count = 0
    fail_count = 0
    samples = []
    for fname in req.files:
        full_path = Path(req.path) / fname
        if not full_path.exists():
            continue

        try:
            data = torch.load(full_path, map_location="cpu", weights_only=False)
        except Exception:
            continue

        # Debug: print structure of first file
        if not debug_printed:
            debug_printed = True
            print(f"[pt_load] first file: {fname}")
            for k, v in data.items():
                t = type(v).__name__
                if hasattr(v, 'shape'):
                    print(f"  key={k!r}  type={t}  shape={list(v.shape)}  dtype={getattr(v, 'dtype', '?')}")
                else:
                    print(f"  key={k!r}  type={t}  value={str(v)[:200]}")

        try:
            # Extract image (use user-specified key, or auto-detect)
            cur_image_key = image_key
            if cur_image_key not in data:
                for k, v in data.items():
                    if isinstance(v, (torch.Tensor, np.ndarray)) and v.ndim == 3:
                        if v.shape[0] in (1, 3, 4) or v.shape[-1] in (1, 3, 4):
                            cur_image_key = k
                            break
                # Auto-find image key
                for k, v in data.items():
                    if isinstance(v, (torch.Tensor, np.ndarray)) and v.ndim == 3:
                        if v.shape[0] in (1, 3, 4) or v.shape[-1] in (1, 3, 4):
                            image_key = k
                            break

            image_tensor = data.get(cur_image_key) if cur_image_key else None
            if image_tensor is None:
                continue

            img_hash = tensor_hash(image_tensor)
            png_name = f"{img_hash}.png"
            png_path = TEMP_ROOT / png_name
            if not png_path.exists():
                save_tensor_as_png(image_tensor, png_name)

            img = tensor_to_pil(image_tensor)
            width, height = img.size

            # Extract annotations
            annotations = []

            # BBox
            bbox_tensor = data.get(req.bbox_key) if req.bbox_key else None
            label_tensor = data.get(req.label_key) if req.label_key else None
            if bbox_tensor is not None:
                bbox_anns = _convert_bbox(bbox_tensor, req.bbox_mode, label_tensor)
                annotations.extend(bbox_anns)

            # Masks
            mask_tensor = data.get(req.mask_key) if req.mask_key else None
            if mask_tensor is not None:
                mask_anns = _convert_mask(mask_tensor)
                for ma in mask_anns:
                    idx = ma["index"]
                    if idx < len(annotations):
                        annotations[idx]["polygons"] = ma["polygons"]
                    else:
                        annotations.append({"polygons": ma["polygons"]})

            # Keypoints
            kpt_tensor = data.get(req.keypoint_key) if req.keypoint_key else None
            if kpt_tensor is not None and annotations:
                kpts = _convert_keypoint(kpt_tensor)
                for i, ann in enumerate(annotations):
                    if i < len(kpts):
                        ann["keypoints"] = kpts[i]

            samples.append({
                "filename": fname,
                "image_url": f"/api/temp/{png_name}",
                "width": width,
                "height": height,
                "annotations": annotations,
            })
            ok_count += 1
        except Exception as e:
            fail_count += 1
            if fail_count <= 3:
                print(f"[pt_load] FAIL {fname}: {e}")
            continue

    print(f"[pt_load] DONE: {ok_count} ok, {fail_count} fail, {len(samples)} total")
    return JSONResponse({"samples": samples})


@app.post("/api/coco/info")
async def coco_info(req: COCOLoadRequest):
    """Get COCO dataset info: categories, image count, annotation count."""
    files_info = _find_coco_files(Path(req.path))
    if files_info is None or files_info["json"] is None:
        raise HTTPException(404, f"No COCO annotation JSON found in {req.path}")

    coco_data = _load_coco_json(files_info["json"])
    categories = coco_data.get("categories", [])
    images = coco_data.get("images", [])
    annotations = coco_data.get("annotations", [])

    return JSONResponse({
        "json_file": str(files_info["json"]),
        "image_dir": str(files_info["image_dir"]) if files_info["image_dir"] else None,
        "num_images": len(images),
        "num_annotations": len(annotations),
        "categories": [{"id": c["id"], "name": c["name"]} for c in categories],
    })


@app.post("/api/coco/load")
async def coco_load(req: COCOLoadRequest):
    """Randomly sample N images from COCO dataset with their annotations."""
    files_info = _find_coco_files(Path(req.path))
    if files_info is None or files_info["json"] is None:
        raise HTTPException(404, f"No COCO annotation JSON found in {req.path}")

    coco_data = _load_coco_json(files_info["json"])
    image_dir = files_info["image_dir"]

    all_images = coco_data.get("images", [])
    all_anns = coco_data.get("annotations", [])
    categories = {c["id"]: c for c in coco_data.get("categories", [])}

    if not all_images:
        raise HTTPException(400, "No images in COCO dataset")

    n = min(req.count, len(all_images))
    selected = random.sample(all_images, n)

    # Build image_id → annotations index
    anns_by_image: dict = {}
    for ann in all_anns:
        img_id = ann["image_id"]
        anns_by_image.setdefault(img_id, []).append(ann)

    samples = []
    for img_info in selected:
        img_id = img_info["id"]
        file_name = img_info.get("file_name", "")
        img_width = img_info.get("width", 0)
        img_height = img_info.get("height", 0)

        # Find image file
        img_path = None
        if image_dir:
            candidate = image_dir / file_name
            if candidate.exists():
                img_path = candidate
            else:
                # Search recursively
                for ext in (".jpg", ".jpeg", ".png", ".bmp"):
                    found = list(image_dir.rglob(f"{Path(file_name).stem}{ext}"))
                    if found:
                        img_path = found[0] if found[0].suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp") else None
                        if img_path:
                            break
                if img_path is None:
                    # Try exact name
                    matches = list(image_dir.rglob(file_name))
                    if matches:
                        img_path = matches[0]

        # Build image URL
        if img_path and img_path.exists():
            # Copy/symlink to temp for serving
            ext = img_path.suffix.lower()
            url_name = f"coco_{img_id}{ext}"
            dest = TEMP_ROOT / url_name
            if not dest.exists():
                shutil.copy2(img_path, dest)
            image_url = f"/api/temp/{url_name}"
        else:
            image_url = None

        annotations = []
        for ann in anns_by_image.get(img_id, []):
            entry = {
                "id": ann.get("id"),
                "category_id": ann.get("category_id"),
                "category_name": categories.get(ann.get("category_id"), {}).get("name", ""),
            }

            # BBox: COCO uses [x, y, w, h] → convert to xyxy
            bbox = ann.get("bbox")
            if bbox and len(bbox) >= 4:
                x, y, w, h = bbox[:4]
                entry["bbox"] = [x, y, x + w, y + h]

            # Segmentation
            seg = ann.get("segmentation")
            if seg:
                if isinstance(seg, dict):
                    # RLE format
                    try:
                        mask = _decode_coco_rle(seg)
                        polygons = _polygons_from_mask(mask)
                        if polygons:
                            entry["polygons"] = polygons
                    except Exception:
                        entry["polygons"] = []  # pycocotools needed
                elif isinstance(seg, list):
                    # Polygon format: list of lists of points
                    entry["polygons"] = seg  # [[x1,y1,x2,y2,...], ...]

            # Area
            entry["area"] = ann.get("area", 0)
            entry["iscrowd"] = bool(ann.get("iscrowd", 0))

            annotations.append(entry)

        samples.append({
            "filename": file_name,
            "image_id": img_id,
            "image_url": image_url,
            "width": img_width,
            "height": img_height,
            "annotations": annotations,
        })

    return JSONResponse({"samples": samples})


@app.get("/api/temp/{filename}")
async def serve_temp(filename: str):
    """Serve a file from the temp directory."""
    filepath = TEMP_ROOT / filename
    if not filepath.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(filepath)


@app.get("/api/file")
async def serve_file(path: str = Query(...)):
    """Serve an arbitrary file from the filesystem (for COCO images, etc.)."""
    p = Path(path)
    if not p.exists():
        raise HTTPException(404, f"File not found: {path}")
    if not p.is_file():
        raise HTTPException(400, "Not a file")
    return FileResponse(p)


# ============================================================================
# Static files & index
# ============================================================================

@app.get("/")
async def index():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")


# ============================================================================
# Entry point
# ============================================================================

def main():
    import webbrowser
    port = 8787

    def _open_browser():
        time.sleep(0.8)
        webbrowser.open(f"http://localhost:{port}")

    import threading
    threading.Thread(target=_open_browser, daemon=True).start()

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")


if __name__ == "__main__":
    main()
