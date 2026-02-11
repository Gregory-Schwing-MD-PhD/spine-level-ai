#!/bin/bash
set -e

DOCKER_USERNAME="go2432"
PROJECT_NAME="spine-level-ai"
VERSION="1.0"

echo "================================================================"
echo "Building Preprocessing Container"
echo "================================================================"

docker build --no-cache \
    -f docker/Dockerfile.preprocessing \
    -t ${DOCKER_USERNAME}/${PROJECT_NAME}-preprocessing:${VERSION} \
    -t ${DOCKER_USERNAME}/${PROJECT_NAME}-preprocessing:latest \
    .

echo "Pushing to Docker Hub..."
docker push ${DOCKER_USERNAME}/${PROJECT_NAME}-preprocessing:${VERSION}
docker push ${DOCKER_USERNAME}/${PROJECT_NAME}-preprocessing:latest

echo "================================================================"
echo "✓ Preprocessing container ready!"
echo "  ${DOCKER_USERNAME}/${PROJECT_NAME}-preprocessing:latest"
echo "================================================================"
