#!/bin/bash
set -e

DOCKER_USERNAME="go2432"
PROJECT_NAME="spineps-lstv"
VERSION="1.0"

echo "================================================================"
echo "Building YOLOv11 Container"
echo "================================================================"

docker build --no-cache \
    -f docker/Dockerfile.yolo \
    -t ${DOCKER_USERNAME}/${PROJECT_NAME}-yolo:${VERSION} \
    -t ${DOCKER_USERNAME}/${PROJECT_NAME}-yolo:latest \
    .

echo "Pushing to Docker Hub..."
docker push ${DOCKER_USERNAME}/${PROJECT_NAME}-yolo:${VERSION}
docker push ${DOCKER_USERNAME}/${PROJECT_NAME}-yolo:latest

echo "================================================================"
echo "✓ YOLOv11 container ready!"
echo "  ${DOCKER_USERNAME}/${PROJECT_NAME}-yolo:latest"
echo "================================================================"
