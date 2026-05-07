#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "========================================="
echo "  LookLabeledData – 数据集可视化浏览器"
echo "========================================="
echo ""

# Check Python
if command -v python3 &>/dev/null; then
  PYTHON=python3
elif command -v python &>/dev/null; then
  PYTHON=python
else
  echo "ERROR: Python not found"
  exit 1
fi

echo "[1/3] 检查 Python 依赖..."
DEPS="fastapi uvicorn torch pillow"
MISSING=""
for dep in $DEPS; do
  if ! $PYTHON -c "import $dep" 2>/dev/null; then
    MISSING="$MISSING $dep"
  fi
done

# Optional: opencv for mask contour extraction
if ! $PYTHON -c "import cv2" 2>/dev/null; then
  echo "  (可选) opencv-python-headless 未安装, 掩码轮廓提取功能将受限"
fi

if [ -n "$MISSING" ]; then
  echo "  缺少依赖:$MISSING"
  echo "  正在安装..."
  $PYTHON -m pip install -q fastapi uvicorn torch pillow opencv-python-headless
fi

echo "[2/3] 清理旧的临时文件..."
rm -rf /tmp/looklabeled_* 2>/dev/null || true

echo "[3/3] 启动服务..."
echo ""
echo "  浏览器将自动打开 http://localhost:8787"
echo "  按 Ctrl+C 退出"
echo ""

$PYTHON app.py
