#!/bin/bash
set -e

echo "================================================================"
echo "Building ALL Containers"
echo "================================================================"

echo ""
echo "[1/3] Building Preprocessing container..."
bash build_preprocessing.sh

echo ""
echo "[2/3] Building SPINEPS container..."
bash build_spineps.sh

echo ""
echo "[3/3] Building YOLOv11 container..."
bash build_yolo.sh

echo ""
echo "================================================================"
echo "✓ All containers built and pushed!"
echo "================================================================"
echo ""
echo "Images created:"
echo "  go2432/spine-level-ai-preprocessing:latest"
echo "  go2432/spine-level-ai-spineps:latest"
echo "  go2432/spine-level-ai-yolo:latest"
echo ""
echo "Pull on HPC:"
echo "  ./setup_containers.sh"
echo "================================================================"
