#!/bin/bash

# This script creates a .env file with default environment variable
# settings for running the surface-dummy-model_DINI model.

# The script takes only one argument: the analysis time to use for
# inference, in ISO8601 format (e.g. 2025-11-05T090000Z).

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <ANALYSIS_TIME>"
    exit 1
fi

ANALYSIS_TIME="$1"
DINI_ZARR="s3://harmonie-zarr/dini/control/${ANALYSIS_TIME}/single_levels.zarr/"
DATASTORE_INPUT_PATHS="danra.danra_surface=${DINI_ZARR},danra.danra_static=${DINI_ZARR}"

cat <<EOF > .env
ANALYSIS_TIME=${ANALYSIS_TIME}
DATASTORE_INPUT_PATHS=${DATASTORE_INPUT_PATHS}
TIME_DIMENSIONS=time
INFERENCE_WORKDIR=/tmp/inference_workdir
EOF

echo "Created .env file with the following contents:"
cat .env
