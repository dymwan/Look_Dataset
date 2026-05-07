# LookLabeledData

A local web-based visualization tool for inspecting training datasets. Browse, filter, and visually verify PyTorch `.pt` files and COCO JSON datasets through an interactive HTML5 Canvas frontend — no uploads, data never leaves your machine.

## Features

- **PyTorch `.pt` support** — auto-detect image/bbox/label/mask/keypoint tensors inside dict-based `.pt` files, with configurable key-to-role mapping
- **COCO JSON support** — browse annotations JSON, render bbox, polygon (flat or nested), and RLE mask overlays
- **deeprs_light compatible** — designed to inspect `.pt` files produced by deeprs_light Dataset pipelines; any dict with tensor values works
- **Samba / network drive friendly** — browse any filesystem path directly, including mounted SMB shares (`/Volumes/...` or `smb://...`)
- **Zero-copy browsing** — images are converted to temporary PNGs on first access and cached; original data is never modified
- **Single-file backend** — `app.py` + static frontend, no framework, no build step, no database

## Quick Start

### 1. Install dependencies

```bash
pip install fastapi uvicorn torch pillow opencv-python-headless
```

`opencv-python-headless` is optional — without it, mask contour extraction falls back gracefully.

### 2. Launch

```bash
python3 app.py --port 8787
```

Or use the convenience script:

```bash
bash start.sh
```

The browser opens automatically at `http://localhost:8787`.

### 3. Open a dataset

Enter a filesystem path in the sidebar and click **浏览** (Browse). This works with:

- Local paths: `/home/user/data/train.pt`
- Samba mounts: `/Volumes/nas-lab/datasets/coco/`
- Any path the Python process can read

## Usage

### PyTorch `.pt` files

1. Switch to the **PyTorch (.pt)** tab
2. Enter or browse to a directory containing `.pt` files
3. The file list shows all `.pt` files in that directory
4. **Auto-detect keys**: click any `.pt` file to scan its tensor keys — the tool suggests roles (image, bbox, label, mask, keypoint) based on shape and dtype heuristics
5. Map keys to roles using the dropdowns (or rely on auto-detection)
6. Select `xyxy` or `xywh` bbox mode depending on your dataset format
7. Click **加载选中文件** (Load selected) to load specific files, or set a sample count and click **随机抽取** (Random sample)

#### Expected `.pt` file structure

The tool expects `.pt` files saved as dicts, matching the deeprs_light convention:

```python
# Example .pt file structure
{
    "img": torch.Tensor,       # shape (3, H, W) or (H, W, 3)
    "bboxes": torch.Tensor,    # shape (N, 4) or (N, 5) — xyxy or xywh
    "labels": torch.Tensor,    # shape (N,)
    "masks": torch.Tensor,     # shape (N, H, W) — binary or uint8
    "keypoints": torch.Tensor, # shape (N, K, 2) or (N, K, 3)
}
```

Keys can be named arbitrarily — the auto-detection heuristics handle most naming conventions. Manual key mapping is available for unusual layouts.

### COCO JSON datasets

1. Switch to the **COCO** tab
2. Enter the root directory containing both the annotation JSON and an `images/` subdirectory (or `train2017/`, `val2017/`, etc.)
3. The tool auto-discovers `instances_*.json` or any JSON with "instance" / "annotation" / "coco" in the filename
4. Set sample count and click **随机抽取** — images are randomly sampled with their annotations

#### Supported COCO directory layouts

```
dataset/
├── instances_train2017.json
└── images/
    ├── 000001.jpg
    └── ...

# Also supported:
dataset/
├── annotations.json
└── train2017/
    └── ...
```

### Canvas controls

- **BBox** — toggle bounding box rectangles with category colors
- **多边形** — toggle polygon/segmentation overlays
- **关键点** — toggle keypoint dots
- **类别标签** — toggle category label text

Hover over annotations on the canvas to see detail (category name, bbox coordinates).

### Samba / network drive usage

Mount your remote dataset share first, then use the mount path in the sidebar:

```bash
# macOS
open smb://nas-server/datasets

# Linux
sudo mount -t cifs //nas-server/datasets /mnt/datasets -o username=user
```

Then enter the mount path (e.g., `/Volumes/datasets/`) in the path bar. All file browsing and loading happens through the local mount — no extra network hop.

## API Overview

| Endpoint | Method | Description |
|---|---|---|
| `/api/browse` | GET | List directory contents (subdirs, .pt files, .json files) |
| `/api/pt/detect` | POST | Scan a `.pt` file and return tensor keys with role suggestions |
| `/api/pt/load` | POST | Load `.pt` files, convert tensors to PNGs, return annotations |
| `/api/coco/info` | POST | Get COCO dataset summary (categories, image/annotation counts) |
| `/api/coco/load` | POST | Random-sample COCO images with annotations |
| `/api/temp/{file}` | GET | Serve cached PNG from temp directory |

## Project Structure

```
LookLabeledData/
├── app.py              # FastAPI backend (single file)
├── requirements.txt    # Python dependencies
├── start.sh            # Launch script with dependency check
└── static/
    ├── index.html      # Frontend UI (inline JS, no framework)
    ├── app.js          # Canvas rendering + API client logic
    └── style.css       # Styles
```
