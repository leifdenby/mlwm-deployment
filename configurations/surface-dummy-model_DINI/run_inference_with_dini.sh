#!/bin/bash

# This script runs the inference container using initial conditions from DINI
# stored on AWS

# The script takes only one argument: the analysis time to use for
# inference, in ISO8601 format (e.g. 2025-11-05T090000Z).

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <ANALYSIS_TIME>"
    exit 1
fi
ANALYSIS_TIME="$1"

DINI_ZARR="s3://harmonie-zarr/dini/control/${ANALYSIS_TIME}/single_levels.zarr/"
DATASTORE_INPUT_PATHS="danra.danra_surface=${DINI_ZARR},danra.danra_static=${DINI_ZARR}"

ANALYSIS_TIME=${ANALYSIS_TIME}
DATASTORE_INPUT_PATHS=${DATASTORE_INPUT_PATHS}
TIME_DIMENSIONS=time

podman run --rm \
  --device /dev/nvidia0 \
  --device /dev/nvidiactl \
  --device /dev/nvidia-uvm \
  --device /dev/nvidia-uvm-tools \
  --device /dev/nvidia-modeset \
  -v /lib/x86_64-linux-gnu/libcuda.so.1:/lib/x86_64-linux-gnu/libcuda.so.1:ro \
  -v /lib/x86_64-linux-gnu/libnvidia-ml.so.1:/lib/x86_64-linux-gnu/libnvidia-ml.so.1:ro \
  -v /lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1:/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1:ro \
  --shm-size=32g \
  -v ./inference_workdir/:/workspace/inference_workdir/ \
  -e DATASTORE_INPUT_PATHS="${DATASTORE_INPUT_PATHS}" \
  -e TIME_DIMENSIONS="${TIME_DIMENSIONS}" \
  -e ANALYSIS_TIME="${ANALYSIS_TIME}" \
  -e FORECAST_DURATION="PT18H" \
  -e NUM_EVAL_STEPS=6 \
  localhost/surface-dummy-model_dini:latest
