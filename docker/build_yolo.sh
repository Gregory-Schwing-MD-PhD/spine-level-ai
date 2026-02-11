#!/bin/bash
set -e

DOCKER_USERNAME="go2432"
PROJECT_NAME="spine-level-ai"
VERSION="1.0"

echo "================================================================"
echo "Building YOLOv11 Container"
echo "================================================================"

docker build \
    -f Dockerfile.yolo \
    -t ${DOCKER_USERNAME}/${PROJECT_NAME}-yolo:latest \
    .

echo "Pushing to Docker Hub..."
docker push ${DOCKER_USERNAME}/${PROJECT_NAME}-yolo:latest

echo "================================================================"
echo "✓ YOLOv11 container ready!"
echo "  ${DOCKER_USERNAME}/${PROJECT_NAME}-yolo:latest"
echo "================================================================"
