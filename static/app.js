// ============================================================================
// LookLabeledData – Frontend Application
// ============================================================================

// ---- State ----
const state = {
  datasetType: "pt",         // "pt" | "coco"
  currentPath: "",
  keys: [],                  // detected .pt keys
  keyMapping: {},            // { image, bbox, label, mask, keypoint }
  bboxMode: "xyxy",
  samples: [],               // loaded sample data
  currentIndex: 0,
  zoom: 1,
  pan: { x: 0, y: 0 },
  dragging: false,
  dragStart: { x: 0, y: 0 },
  layerVis: {
    bbox: true,
    polygon: true,
    keypoint: true,
    label: true,
  },
  hoveredAnnIdx: -1,
};

// ---- DOM refs ----
const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

const dom = {
  datasetTabs: $$("#dataset-type-tabs .tab"),
  pathInput: $("#path-input"),
  btnBrowse: $("#btn-browse"),
  breadcrumb: $("#breadcrumb"),
  fileList: $("#file-list"),
  keymapPanel: $("#keymap-panel"),
  keymapLoading: $("#keymap-loading"),
  keymapContent: $("#keymap-content"),
  keymapGrid: $("#keymap-grid"),
  bboxModeRow: $("#bbox-mode-row"),
  bboxMode: $("#bbox-mode"),
  detectStatus: $("#detect-status"),
  btnLoadSelected: $("#btn-load-selected"),
  btnRandom: $("#btn-random"),
  sampleCount: $("#sample-count"),
  sampleStatus: $("#sample-status"),
  showBbox: $("#show-bbox"),
  showPolygon: $("#show-polygon"),
  showKeypoint: $("#show-keypoint"),
  showLabel: $("#show-label"),
  canvasContainer: $("#canvas-container"),
  imageWrapper: $("#image-wrapper"),
  mainImage: $("#main-image"),
  overlayCanvas: $("#overlay-canvas"),
  noDataMsg: $("#no-data-msg"),
  infoBar: $("#info-bar"),
  infoFilename: $("#info-filename"),
  infoDims: $("#info-dims"),
  infoAnnots: $("#info-annots"),
  navBar: $("#nav-bar"),
  btnPrev: $("#btn-prev"),
  btnNext: $("#btn-next"),
  navIndex: $("#nav-index"),
  thumbnailStrip: $("#thumbnail-strip"),
};

// ---------------------------------------------------------------------------
// API helpers
// ---------------------------------------------------------------------------

async function apiGet(url) {
  const r = await fetch(url);
  return r.json();
}

async function apiPost(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(err.detail || `HTTP ${r.status}`);
  }
  return r.json();
}

// ---------------------------------------------------------------------------
// Dataset type toggle
// ---------------------------------------------------------------------------

dom.datasetTabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    dom.datasetTabs.forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    state.datasetType = tab.dataset.type;
    updateUIForDatasetType();
  });
});

function updateUIForDatasetType() {
  const isPT = state.datasetType === "pt";
  dom.keymapPanel.style.display = isPT ? "" : "none";
  dom.bboxModeRow.style.display = "none";
  dom.btnLoadSelected.style.display = "none";
  // Clear previous state
  state.keys = [];
  state.samples = [];
  resetView();
}

// ---------------------------------------------------------------------------
// Path browsing
// ---------------------------------------------------------------------------

dom.btnBrowse.addEventListener("click", () => browsePath(state.pathInput.value.trim()));
dom.pathInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") browsePath(state.pathInput.value.trim());
});

async function browsePath(path) {
  if (!path) return;
  state.currentPath = path;
  dom.pathInput.value = path;
  dom.fileList.innerHTML = '<span class="spinner"></span> 加载中...';

  try {
    const data = await apiGet(`/api/browse?path=${encodeURIComponent(path)}`);
    renderBreadcrumb(data);
    renderFileList(data);

    if (state.datasetType === "pt" && data.pt_files.length > 0) {
      // Auto-detect first .pt file
      autoDetectFirst(data.pt_files[0]);
    } else if (state.datasetType === "coco" && data.json_files.length > 0) {
      dom.sampleStatus.textContent = `发现 ${data.json_files.length} 个 JSON 文件`;
    }
  } catch (e) {
    dom.fileList.innerHTML = `<p class="hint" style="color:#e94560;">错误: ${e.message}</p>`;
  }
}

function renderBreadcrumb(data) {
  const parts = data.path.split("/").filter(Boolean);
  let html = '<span data-path="/">/</span>';
  let cumulative = "";
  for (const part of parts) {
    cumulative += "/" + part;
    html += `<span data-path="${cumulative}">${part}</span> / `;
  }
  dom.breadcrumb.innerHTML = html;
  dom.breadcrumb.querySelectorAll("span").forEach((el) => {
    el.addEventListener("click", () => browsePath(el.dataset.path));
  });
}

function renderFileList(data) {
  if (data.error) {
    dom.fileList.innerHTML = `<p class="hint" style="color:#e94560;">${data.error}</p>`;
    return;
  }

  let html = "";

  if (data.parent !== data.path) {
    html += `<div class="item" data-path="${data.parent}">
      <span class="icon">📁</span> ../
    </div>`;
  }

  for (const d of data.dirs) {
    html += `<div class="item dir" data-path="${d.path}">
      <span class="icon">📁</span> ${d.name}
    </div>`;
  }

  const showFiles = state.datasetType === "pt" ? data.pt_files : data.files;
  const icon = state.datasetType === "pt" ? "📄" : "🖼";

  for (const f of showFiles) {
    html += `<div class="item file" data-path="${f.path}">
      <span class="icon">${icon}</span> ${f.name}
    </div>`;
  }

  dom.fileList.innerHTML = html || '<p class="hint">空目录</p>';

  // Click handlers
  dom.fileList.querySelectorAll(".item.dir").forEach((el) => {
    el.addEventListener("click", () => browsePath(el.dataset.path));
  });
  dom.fileList.querySelectorAll(".item.file").forEach((el) => {
    el.addEventListener("click", () => {
      dom.fileList.querySelectorAll(".item").forEach((i) => i.classList.remove("selected"));
      el.classList.add("selected");
      if (state.datasetType === "pt") {
        detectKeys(el.dataset.path);
      }
    });
  });
}

// ---------------------------------------------------------------------------
// PT Key detection
// ---------------------------------------------------------------------------

function autoDetectFirst(ptFile) {
  detectKeys(ptFile.path);
}

async function detectKeys(filePath) {
  dom.keymapLoading.style.display = "";
  dom.keymapContent.style.display = "none";
  dom.detectStatus.textContent = "";

  const dir = filePath.substring(0, filePath.lastIndexOf("/"));
  const file = filePath.substring(filePath.lastIndexOf("/") + 1);

  try {
    const data = await apiPost("/api/pt/detect", { path: dir, file });
    state.keys = data.keys;
    state.currentPath = dir;
    renderKeyMapping(data.keys);
    dom.keymapLoading.style.display = "none";
    dom.keymapContent.style.display = "";
    dom.detectStatus.textContent = `已检测: ${file}`;
  } catch (e) {
    dom.keymapLoading.style.display = "none";
    dom.detectStatus.textContent = `检测失败: ${e.message}`;
  }
}

// Silent version: detect keys without UI changes, for use during random sample
async function detectKeysSilent(filePath) {
  const dir = filePath.substring(0, filePath.lastIndexOf("/"));
  const file = filePath.substring(filePath.lastIndexOf("/") + 1);
  const data = await apiPost("/api/pt/detect", { path: dir, file });
  state.keys = data.keys;
  // Auto-assign mappings from suggestions
  const autoMap = {};
  for (const k of data.keys) {
    if (k.suggested_role === "image" && !autoMap.image) autoMap.image = k.name;
    if ((k.suggested_role === "bbox" || k.suggested_role === "bbox_label") && !autoMap.bbox) autoMap.bbox = k.name;
    if (k.suggested_role === "label" && !autoMap.label) autoMap.label = k.name;
    if (k.suggested_role === "mask" && !autoMap.mask) autoMap.mask = k.name;
    if (k.suggested_role === "keypoint" && !autoMap.keypoint) autoMap.keypoint = k.name;
  }
  state.keyMapping = autoMap;
}

function renderKeyMapping(keys) {
  const roles = [
    { value: "", label: "-- 忽略 --" },
    { value: "image", label: "图像" },
    { value: "bbox", label: "BBox" },
    { value: "bbox_label", label: "BBox+类别" },
    { value: "label", label: "类别标签" },
    { value: "mask", label: "掩码" },
    { value: "keypoint", label: "关键点" },
  ];

  // Auto-assign mapping based on suggestions
  const autoMap = {};
  for (const k of keys) {
    if (k.suggested_role === "image" && !autoMap.image) autoMap.image = k.name;
    if ((k.suggested_role === "bbox" || k.suggested_role === "bbox_label") && !autoMap.bbox) autoMap.bbox = k.name;
    if (k.suggested_role === "label" && !autoMap.label) autoMap.label = k.name;
    if (k.suggested_role === "mask" && !autoMap.mask) autoMap.mask = k.name;
    if (k.suggested_role === "keypoint" && !autoMap.keypoint) autoMap.keypoint = k.name;
  }
  state.keyMapping = autoMap;

  let html = '<div class="keymap-row header"><span class="name">键名</span><span class="shape">形状/类型</span><span>角色</span></div>';
  for (const k of keys) {
    let shapeStr = k.shape ? k.shape.join(" × ") : (k.len !== undefined ? `list[${k.len}]` : k.type);
    if (k.dtype) shapeStr += ` (${k.dtype})`;
    if (k.element_shape) shapeStr += ` of ${k.element_shape.join("×")}`;

    const currentRole = Object.entries(autoMap).find(([, v]) => v === k.name)?.[0] || "";
    const opts = roles.map((r) => {
      const sel = r.value === currentRole ? " selected" : "";
      return `<option value="${r.value}"${sel}>${r.label}</option>`;
    }).join("");

    html += `<div class="keymap-row">
      <span class="name" title="${k.name}">${k.name}</span>
      <span class="shape">${shapeStr}</span>
      <select data-key="${k.name}">${opts}</select>
    </div>`;
  }

  dom.keymapGrid.innerHTML = html;

  // Show bbox mode if bbox is mapped
  const hasBbox = autoMap.bbox || autoMap.label;
  dom.bboxModeRow.style.display = hasBbox ? "flex" : "none";

  // Show load button
  dom.btnLoadSelected.style.display = "";

  // Listen for mapping changes
  dom.keymapGrid.querySelectorAll("select").forEach((sel) => {
    sel.addEventListener("change", () => {
      const keyName = sel.dataset.key;
      const role = sel.value;
      // Clear previous mapping
      for (const [r, v] of Object.entries(state.keyMapping)) {
        if (v === keyName) delete state.keyMapping[r];
      }
      if (role && role !== "bbox_label") {
        state.keyMapping[role] = keyName;
      } else if (role === "bbox_label") {
        state.keyMapping.bbox = keyName;
      }
      const hasBbox = state.keyMapping.bbox;
      dom.bboxModeRow.style.display = hasBbox ? "flex" : "none";
    });
  });
}

// ---------------------------------------------------------------------------
// Load / Random Sample
// ---------------------------------------------------------------------------

dom.btnLoadSelected.addEventListener("click", () => {
  const selected = dom.fileList.querySelectorAll(".item.file.selected");
  if (selected.length === 0) return;
  const files = [];
  selected.forEach((el) => {
    const path = el.dataset.path;
    files.push(path.substring(path.lastIndexOf("/") + 1));
  });
  loadSamples(files);
});

dom.btnRandom.addEventListener("click", () => randomSample());

async function randomSample() {
  const path = state.currentPath || state.pathInput.value.trim();
  if (!path) {
    dom.sampleStatus.textContent = "请先输入数据路径";
    return;
  }
  state.currentPath = path;

  dom.btnRandom.disabled = true;
  dom.btnRandom.textContent = "加载中...";
  dom.sampleStatus.textContent = "";

  try {
    if (state.datasetType === "pt") {
      // List pt files, pick random N
      const data = await apiGet(`/api/browse?path=${encodeURIComponent(path)}`);
      const files = data.pt_files.map((f) => f.name);
      if (files.length === 0) {
        dom.sampleStatus.textContent = "没有找到 .pt 文件";
        return;
      }
      // Auto-detect keys from first file if not yet done
      if (state.keys.length === 0) {
        try {
          await detectKeysSilent(path + "/" + files[0]);
        } catch (e) { /* continue without key mapping */ }
      }
      const count = Math.min(parseInt(dom.sampleCount.value) || 10, files.length);
      const selected = shuffleSlice(files, count);
      await loadSamples(selected);
      dom.sampleStatus.textContent = `已加载 ${selected.length} / ${files.length} 个文件`;
    } else {
      // COCO
      try {
        await apiPost("/api/coco/info", { path });
      } catch {
        dom.sampleStatus.textContent = "未找到 COCO JSON 文件，尝试直接加载...";
      }
      const count = parseInt(dom.sampleCount.value) || 10;
      const data = await apiPost("/api/coco/load", { path, count });
      state.samples = data.samples;
      state.currentIndex = 0;
      renderSamples();
    }
  } catch (e) {
    dom.sampleStatus.textContent = `错误: ${e.message}`;
  } finally {
    dom.btnRandom.disabled = false;
    dom.btnRandom.textContent = "随机抽取";
  }
}

function shuffleSlice(arr, n) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a.slice(0, n);
}

async function loadSamples(files) {
  if (!state.currentPath) return;
  const keyMapping = { ...state.keyMapping };

  // Ensure image key is set – auto-detect if needed
  if (!keyMapping.image) {
    for (const k of state.keys) {
      if (k.suggested_role === "image") {
        keyMapping.image = k.name;
        break;
      }
    }
  }

  dom.sampleStatus.textContent = "加载中...";
  try {
    const data = await apiPost("/api/pt/load", {
      path: state.currentPath,
      files,
      image_key: keyMapping.image || "",
      bbox_key: keyMapping.bbox || "",
      label_key: keyMapping.label || "",
      mask_key: keyMapping.mask || "",
      keypoint_key: keyMapping.keypoint || "",
      bbox_mode: state.bboxMode,
    });
    state.samples = data.samples.filter((s) => s.image_url);
    state.currentIndex = 0;
    renderSamples();
    dom.sampleStatus.textContent = `已加载 ${state.samples.length} 个文件`;
  } catch (e) {
    dom.sampleStatus.textContent = `错误: ${e.message}`;
  }
}

// ---------------------------------------------------------------------------
// Render samples: main image + canvas overlay + thumbnails
// ---------------------------------------------------------------------------

function renderSamples() {
  if (state.samples.length === 0) {
    resetView();
    return;
  }

  dom.noDataMsg.style.display = "none";
  dom.navBar.style.display = "flex";
  dom.infoBar.style.display = "flex";
  dom.thumbnailStrip.style.display = "flex";

  state.zoom = 1;
  state.pan = { x: 0, y: 0 };
  showSample(state.currentIndex);
  renderThumbnails();
}

function resetView() {
  state.samples = [];
  state.currentIndex = 0;
  dom.mainImage.style.display = "none";
  dom.mainImage.src = "";
  dom.overlayCanvas.style.display = "none";
  dom.noDataMsg.style.display = "";
  dom.navBar.style.display = "none";
  dom.infoBar.style.display = "none";
  dom.thumbnailStrip.style.display = "none";
  dom.thumbnailStrip.innerHTML = "";
  state.zoom = 1;
  state.pan = { x: 0, y: 0 };
}

function showSample(idx) {
  if (idx < 0 || idx >= state.samples.length) return;
  state.currentIndex = idx;
  const sample = state.samples[idx];

  dom.mainImage.style.display = "";
  dom.mainImage.src = sample.image_url;
  dom.overlayCanvas.style.display = "";

  dom.infoFilename.textContent = sample.filename;
  dom.infoDims.textContent = `${sample.width} × ${sample.height}`;
  const nAnnot = sample.annotations ? sample.annotations.length : 0;
  dom.infoAnnots.textContent = `${nAnnot} 个标注`;

  dom.navIndex.textContent = `${idx + 1} / ${state.samples.length}`;

  // Update thumbnail highlight
  dom.thumbnailStrip.querySelectorAll(".thumb").forEach((t, i) => {
    t.classList.toggle("active", i === idx);
  });

  // Wait for image to load before drawing overlay
  dom.mainImage.onload = () => {
    sizeCanvasToImage();
    drawOverlay();
  };
  dom.mainImage.onerror = () => {
    dom.infoFilename.textContent = sample.filename + " (图片加载失败)";
  };
}

function sizeCanvasToImage() {
  const img = dom.mainImage;
  const canvas = dom.overlayCanvas;
  const wrapper = dom.imageWrapper;

  const natW = img.naturalWidth;
  const natH = img.naturalHeight;
  if (!natW || !natH) return;

  // Scale to fit container while preserving aspect ratio
  const maxW = dom.canvasContainer.clientWidth * 0.92;
  const maxH = dom.canvasContainer.clientHeight * 0.75;
  const ratio = Math.min(maxW / natW, maxH / natH, 1.0);
  const dispW = Math.round(natW * ratio);
  const dispH = Math.round(natH * ratio);

  wrapper.style.width = dispW + "px";
  wrapper.style.height = dispH + "px";
  img.style.width = dispW + "px";
  img.style.height = dispH + "px";

  canvas.width = dispW;
  canvas.height = dispH;
  canvas.style.width = dispW + "px";
  canvas.style.height = dispH + "px";
  canvas.style.left = "0";
  canvas.style.top = "0";
}

function renderThumbnails() {
  let html = "";
  for (let i = 0; i < state.samples.length; i++) {
    const cls = i === state.currentIndex ? " active" : "";
    html += `<img class="thumb${cls}" src="${state.samples[i].image_url}" data-idx="${i}" loading="lazy">`;
  }
  dom.thumbnailStrip.innerHTML = html;

  dom.thumbnailStrip.querySelectorAll(".thumb").forEach((t) => {
    t.addEventListener("click", () => showSample(parseInt(t.dataset.idx)));
  });
}

// ---------------------------------------------------------------------------
// Canvas overlay drawing
// ---------------------------------------------------------------------------

// Category color palette
const CAT_COLORS = [
  "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
  "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
  "#469990", "#e6beff", "#9a6324", "#800000", "#aaffc3",
  "#808000", "#ffd8b1", "#000075", "#a9a9a9", "#000000",
  "#00ffff", "#ff00ff", "#ffff00", "#ff8000", "#80ff00",
];

function getColor(catId, alpha = 1) {
  if (catId === undefined || catId === null) catId = 0;
  const hex = CAT_COLORS[catId % CAT_COLORS.length];
  if (alpha === 1) return hex;
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

function drawOverlay() {
  const sample = state.samples[state.currentIndex];
  if (!sample || !sample.annotations) return;

  const canvas = dom.overlayCanvas;
  const ctx = canvas.getContext("2d");
  const img = dom.mainImage;
  const scaleX = canvas.width / sample.width;
  const scaleY = canvas.height / sample.height;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  for (let i = 0; i < sample.annotations.length; i++) {
    const ann = sample.annotations[i];
    const catId = ann.category_id ?? ann.label ?? 0;
    const color = getColor(catId);
    const isHovered = i === state.hoveredAnnIdx;
    const lineWidth = isHovered ? 3 : 1.5;
    const fillAlpha = isHovered ? 0.35 : 0.15;

    // Draw polygons / segmentation
    if (state.layerVis.polygon && ann.polygons) {
      for (const poly of ann.polygons) {
        if (poly.length < 3) continue;
        ctx.beginPath();
        // poly can be [[x,y], [x,y], ...] or [x1,y1,x2,y2,...]
        if (Array.isArray(poly[0])) {
          ctx.moveTo(poly[0][0] * scaleX, poly[0][1] * scaleY);
          for (let j = 1; j < poly.length; j++) {
            ctx.lineTo(poly[j][0] * scaleX, poly[j][1] * scaleY);
          }
        } else {
          ctx.moveTo(poly[0] * scaleX, poly[1] * scaleY);
          for (let j = 2; j < poly.length; j += 2) {
            ctx.lineTo(poly[j] * scaleX, poly[j + 1] * scaleY);
          }
        }
        ctx.closePath();
        ctx.fillStyle = getColor(catId, fillAlpha);
        ctx.fill();
        ctx.strokeStyle = color;
        ctx.lineWidth = lineWidth;
        ctx.stroke();
      }
    }

    // Draw bbox
    if (state.layerVis.bbox && ann.bbox) {
      const [x1, y1, x2, y2] = ann.bbox;
      const sx1 = x1 * scaleX, sy1 = y1 * scaleY;
      const sw = (x2 - x1) * scaleX, sh = (y2 - y1) * scaleY;

      ctx.strokeStyle = isHovered ? "#fff" : color;
      ctx.lineWidth = lineWidth;
      ctx.strokeRect(sx1, sy1, sw, sh);

      if (state.layerVis.label) {
        const labelText = ann.category_name || ann.label || `#${catId}`;
        if (labelText !== undefined) {
          const fontSize = Math.max(10, Math.min(14, sw * 0.3));
          ctx.font = `${isHovered ? "bold " : ""}${fontSize}px ${getComputedStyle(document.body).fontFamily}`;
          const metrics = ctx.measureText(String(labelText));
          const pad = 3;
          const labelH = fontSize + pad * 2;
          ctx.fillStyle = getColor(catId, 0.85);
          ctx.fillRect(sx1, sy1 - labelH, metrics.width + pad * 2, labelH);
          ctx.fillStyle = "#fff";
          ctx.fillText(String(labelText), sx1 + pad, sy1 - pad);
        }
      }
    }

    // Draw keypoints
    if (state.layerVis.keypoint && ann.keypoints) {
      const kpts = Array.isArray(ann.keypoints) ? ann.keypoints : [ann.keypoints];
      for (const pt of kpts) {
        if (Array.isArray(pt) && pt.length >= 2) {
          ctx.beginPath();
          ctx.arc(pt[0] * scaleX, pt[1] * scaleY, isHovered ? 5 : 3, 0, Math.PI * 2);
          ctx.fillStyle = color;
          ctx.fill();
          ctx.strokeStyle = "#fff";
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Navigation
// ---------------------------------------------------------------------------

dom.btnPrev.addEventListener("click", () => {
  if (state.currentIndex > 0) showSample(state.currentIndex - 1);
});
dom.btnNext.addEventListener("click", () => {
  if (state.currentIndex < state.samples.length - 1) showSample(state.currentIndex + 1);
});

document.addEventListener("keydown", (e) => {
  if (state.samples.length === 0) return;
  if (e.key === "ArrowLeft") {
    e.preventDefault();
    if (state.currentIndex > 0) showSample(state.currentIndex - 1);
  } else if (e.key === "ArrowRight") {
    e.preventDefault();
    if (state.currentIndex < state.samples.length - 1) showSample(state.currentIndex + 1);
  } else if (e.key === "r" && !e.ctrlKey && !e.metaKey) {
    // 'r' to random sample when not in input
    if (document.activeElement === document.body || document.activeElement === dom.canvasContainer) {
      randomSample();
    }
  }
});

// ---------------------------------------------------------------------------
// Zoom & Pan (mouse wheel + drag)
// ---------------------------------------------------------------------------

dom.canvasContainer.addEventListener("wheel", (e) => {
  if (!dom.mainImage.src || state.samples.length === 0) return;
  e.preventDefault();
  const delta = e.deltaY > 0 ? 0.9 : 1.1;
  const newZoom = Math.max(0.1, Math.min(10, state.zoom * delta));
  state.zoom = newZoom;
  applyTransform();
});

dom.imageWrapper.addEventListener("mousedown", (e) => {
  if (e.button !== 0) return; // left button only
  state.dragging = true;
  state.dragStart = { x: e.clientX - state.pan.x, y: e.clientY - state.pan.y };
  dom.imageWrapper.style.cursor = "grabbing";
});

window.addEventListener("mousemove", (e) => {
  if (!state.dragging) return;
  state.pan.x = e.clientX - state.dragStart.x;
  state.pan.y = e.clientY - state.dragStart.y;
  applyTransform();
});

window.addEventListener("mouseup", () => {
  state.dragging = false;
  dom.imageWrapper.style.cursor = state.samples.length > 0 ? "grab" : "";
});

function applyTransform() {
  dom.imageWrapper.style.transform = `translate(${state.pan.x}px, ${state.pan.y}px) scale(${state.zoom})`;
  dom.imageWrapper.style.transformOrigin = "center center";
}

// Double-click to reset zoom
dom.imageWrapper.addEventListener("dblclick", () => {
  state.zoom = 1;
  state.pan = { x: 0, y: 0 };
  applyTransform();
});

// ---------------------------------------------------------------------------
// Layer visibility toggles
// ---------------------------------------------------------------------------

dom.showBbox.addEventListener("change", () => {
  state.layerVis.bbox = dom.showBbox.checked;
  drawOverlay();
});
dom.showPolygon.addEventListener("change", () => {
  state.layerVis.polygon = dom.showPolygon.checked;
  drawOverlay();
});
dom.showKeypoint.addEventListener("change", () => {
  state.layerVis.keypoint = dom.showKeypoint.checked;
  drawOverlay();
});
dom.showLabel.addEventListener("change", () => {
  state.layerVis.label = dom.showLabel.checked;
  drawOverlay();
});

// ---------------------------------------------------------------------------
// Hover on canvas to highlight annotations
// ---------------------------------------------------------------------------

dom.overlayCanvas.addEventListener("mousemove", (e) => {
  const sample = state.samples[state.currentIndex];
  if (!sample || !sample.annotations) return;

  const canvas = dom.overlayCanvas;
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;
  const scaleX = canvas.width / sample.width;
  const scaleY = canvas.height / sample.height;

  let found = -1;
  // Check bboxes first (fast)
  for (let i = sample.annotations.length - 1; i >= 0; i--) {
    const ann = sample.annotations[i];
    if (ann.bbox) {
      const [x1, y1, x2, y2] = ann.bbox;
      if (mx >= x1 * scaleX && mx <= x2 * scaleX && my >= y1 * scaleY && my <= y2 * scaleY) {
        found = i;
        break;
      }
    }
  }

  if (found !== state.hoveredAnnIdx) {
    state.hoveredAnnIdx = found;
    dom.overlayCanvas.style.cursor = found >= 0 ? "pointer" : "crosshair";
    drawOverlay();
  }
});

dom.overlayCanvas.addEventListener("mouseleave", () => {
  if (state.hoveredAnnIdx !== -1) {
    state.hoveredAnnIdx = -1;
    drawOverlay();
  }
});

// ---------------------------------------------------------------------------
// BBox mode change
// ---------------------------------------------------------------------------

dom.bboxMode.addEventListener("change", () => {
  state.bboxMode = dom.bboxMode.value;
});

// ---------------------------------------------------------------------------
// Window resize → re-draw overlay
// ---------------------------------------------------------------------------

window.addEventListener("resize", () => {
  if (state.samples.length > 0) {
    // Re-size canvas after a short delay to let layout settle
    setTimeout(() => {
      sizeCanvasToImage();
      drawOverlay();
    }, 200);
  }
});

// Use ResizeObserver for more precise tracking
if (window.ResizeObserver) {
  const ro = new ResizeObserver(() => {
    if (state.samples.length > 0) {
      sizeCanvasToImage();
      drawOverlay();
    }
  });
  ro.observe(dom.canvasContainer);
}

// ---------------------------------------------------------------------------
// Prefill path from URL hash
// ---------------------------------------------------------------------------

window.addEventListener("DOMContentLoaded", () => {
  const hash = window.location.hash.slice(1);
  if (hash) {
    dom.pathInput.value = decodeURIComponent(hash);
  }
});
