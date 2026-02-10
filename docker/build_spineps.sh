#!/bin/bash
# Build and push SPINEPS container

set -e

DOCKER_USERNAME="go2432"
PROJECT_NAME="spineps-lstv"

echo "Building SPINEPS container ..."

docker build \
    -f docker/Dockerfile.spineps \
    -t ${DOCKER_USERNAME}/${PROJECT_NAME}:latest \
    .

echo "Pushing to Docker Hub..."
docker push ${DOCKER_USERNAME}/${PROJECT_NAME}:latest

echo "Done! Container: ${DOCKER_USERNAME}/${PROJECT_NAME}:latest"
