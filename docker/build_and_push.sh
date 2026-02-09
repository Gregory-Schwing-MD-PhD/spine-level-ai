#!/bin/bash
# Build and push Docker containers to Docker Hub
# Run this on a machine with Docker installed (not on HPC)

set -e

# Configuration
DOCKER_USERNAME="${DOCKER_USERNAME:-go2432}"  # Change to your Docker Hub username
PROJECT_NAME="spine-level-ai"
VERSION="1.0"

echo "================================================================"
echo "Building Docker Containers for ${PROJECT_NAME}"
echo "================================================================"

# Check if logged into Docker Hub
if ! docker info > /dev/null 2>&1; then
    echo "Docker is not running or you don't have permission"
    echo "Please start Docker and ensure you have access"
    exit 1
fi

echo "Checking Docker Hub login status..."
if ! docker info | grep -q "Username"; then
    echo "Not logged into Docker Hub. Logging in..."
    docker login
else
    echo "Already logged into Docker Hub"
fi

# Build preprocessing container
echo ""
echo "================================================================"
echo "Building Preprocessing Container"
echo "================================================================"
docker build \
    -f docker/Dockerfile.preprocessing \
    -t ${DOCKER_USERNAME}/${PROJECT_NAME}-preprocessing:${VERSION} \
    -t ${DOCKER_USERNAME}/${PROJECT_NAME}-preprocessing:latest \
    .

echo "Pushing preprocessing container to Docker Hub..."
docker push ${DOCKER_USERNAME}/${PROJECT_NAME}-preprocessing:${VERSION}
docker push ${DOCKER_USERNAME}/${PROJECT_NAME}-preprocessing:latest

# Build training container
echo ""
echo "================================================================"
echo "Building Training Container"
echo "================================================================"
docker build \
    -f docker/Dockerfile.training \
    -t ${DOCKER_USERNAME}/${PROJECT_NAME}-training:${VERSION} \
    -t ${DOCKER_USERNAME}/${PROJECT_NAME}-training:latest \
    .

echo "Pushing training container to Docker Hub..."
docker push ${DOCKER_USERNAME}/${PROJECT_NAME}-training:${VERSION}
docker push ${DOCKER_USERNAME}/${PROJECT_NAME}-training:latest

echo ""
echo "================================================================"
echo "Build Complete!"
echo "================================================================"
echo ""
echo "Containers pushed to Docker Hub:"
echo "  - ${DOCKER_USERNAME}/${PROJECT_NAME}-preprocessing:latest"
echo "  - ${DOCKER_USERNAME}/${PROJECT_NAME}-training:latest"
echo ""
echo "To pull on HPC:"
echo "  singularity pull docker://${DOCKER_USERNAME}/${PROJECT_NAME}-preprocessing:latest"
echo "  singularity pull docker://${DOCKER_USERNAME}/${PROJECT_NAME}-training:latest"
echo ""
echo "Or use directly in SLURM scripts with:"
echo "  CONTAINER=\"docker://${DOCKER_USERNAME}/${PROJECT_NAME}-training:latest\""
echo ""
