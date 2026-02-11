#!/bin/bash
export SPINEPS_SEGMENTOR_MODELS=/app/models
export SPINEPS_ENVIRONMENT_DIR=/app/models
mkdir -p /app/models
exec spineps "$@"
