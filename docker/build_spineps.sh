#!/bin/bash
set -e

DOCKER_USERNAME="go2432"
PROJECT_NAME="spine-level-ai"
VERSION="1.0"

echo "================================================================"
echo "Building SPINEPS Container"
echo "================================================================"

docker build \
    -f docker/Dockerfile.spineps \
    -t ${DOCKER_USERNAME}/${PROJECT_NAME}-spineps:latest \
    .

echo "Pushing to Docker Hub..."
docker push ${DOCKER_USERNAME}/${PROJECT_NAME}-spineps:latest

echo "================================================================"
echo "✓ SPINEPS container ready!"
echo "  ${DOCKER_USERNAME}/${PROJECT_NAME}-spineps:latest"
echo "================================================================"
