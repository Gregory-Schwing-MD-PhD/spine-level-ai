#!/bin/bash
# Build and push SPINEPS container

set -e

DOCKER_USERNAME="go2432"
PROJECT_NAME="spineps-lstv"
VERSION="1.0"

echo "Building SPINEPS container (NO CACHE)..."

docker build --no-cache \
    -f docker/Dockerfile.spineps \
    -t ${DOCKER_USERNAME}/${PROJECT_NAME}:${VERSION} \
    -t ${DOCKER_USERNAME}/${PROJECT_NAME}:latest \
    .

echo "Pushing to Docker Hub..."
docker push ${DOCKER_USERNAME}/${PROJECT_NAME}:${VERSION}
docker push ${DOCKER_USERNAME}/${PROJECT_NAME}:latest

echo "Done! Container: ${DOCKER_USERNAME}/${PROJECT_NAME}:latest"
