#!/bin/bash
#
# FIXED SPINEPS WRAPPER v2.0
# Properly executes SPINEPS by calling Python module instead of broken CLI
#

set -e

# Set environment variables
export SPINEPS_SEGMENTOR_MODELS=${SPINEPS_SEGMENTOR_MODELS:-/app/models}
export SPINEPS_ENVIRONMENT_DIR=${SPINEPS_ENVIRONMENT_DIR:-/app/models}

# Ensure model directory exists
mkdir -p "${SPINEPS_SEGMENTOR_MODELS}"

# CRITICAL FIX: The 'spineps' CLI is broken in the Docker image
# We need to call it through Python module instead
exec python -m spineps.entrypoint "$@"
