#!/bin/bash
# Wrapper to set SPINEPS environment before import

# Force SPINEPS to use /app/models
export SPINEPS_SEGMENTOR_MODELS=/app/models
export SPINEPS_ENVIRONMENT_DIR=/app/models

# Create the directory (writable location)
mkdir -p /app/models

# Run spineps with all arguments passed to this script
exec spineps "$@"
