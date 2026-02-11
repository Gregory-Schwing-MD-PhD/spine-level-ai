#!/bin/bash
# Pull all containers on HPC

set -e

CACHE_DIR="${HOME}/singularity_cache"
mkdir -p $CACHE_DIR

echo "================================================================"
echo "Pulling All LSTV Detection Containers"
echo "================================================================"

echo ""
echo "[1/3] Pulling Preprocessing container..."
singularity pull --force \
    ${CACHE_DIR}/spine-level-ai-preprocessing.sif \
    docker://go2432/spine-level-ai-preprocessing:latest

echo "✓ Preprocessing container ready: ${CACHE_DIR}/spine-level-ai-preprocessing.sif"

echo ""
echo "[2/3] Pulling SPINEPS container..."
singularity pull --force \
    ${CACHE_DIR}/spineps.sif \
    docker://go2432/spineps-lstv-spineps:latest

echo "✓ SPINEPS container ready: ${CACHE_DIR}/spineps.sif"

echo ""
echo "[3/3] Pulling YOLOv11 container..."
singularity pull --force \
    ${CACHE_DIR}/yolo.sif \
    docker://go2432/spineps-lstv-yolo:latest

echo "✓ YOLOv11 container ready: ${CACHE_DIR}/yolo.sif"

echo ""
echo "================================================================"
echo "All containers ready!"
echo "================================================================"
echo "Preprocessing: ${CACHE_DIR}/spine-level-ai-preprocessing.sif"
echo "SPINEPS:       ${CACHE_DIR}/spineps.sif"
echo "YOLOv11:       ${CACHE_DIR}/yolo.sif"
echo "================================================================"
