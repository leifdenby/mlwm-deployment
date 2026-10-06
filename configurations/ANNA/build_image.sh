#!/bin/bash

# Container application (defuault to podman if not set)
CONTAINER_APP=${CONTAINER_APP:-podman}

# Configuration
MLWM_LOG_LEVEL=DEBUG
MLWM_IMAGE_NAME="anna:latest"

HTTP_PROXY=""
HTTPS_PROXY=""

# Set MLWM_PULL_PROXY before running this script, e.g.:
#   export MLWM_PULL_PROXY="your.proxy.server:port"
if [ -z "$MLWM_PULL_PROXY" ]; then
	echo "Info: MLWM_PULL_PROXY is not set. Using public DockerHub."
	MLWM_PULL_PROXY=""
    CR_URL="docker.io"
else
	echo "Info: Using proxy $MLWM_PULL_PROXY and internal DockerHub."
    CR_URL="dockerhub.dmi.dk"
fi

# if we're on ARM architecture, use the ARM base image
if [ "$(uname -m)" = "aarch64" ]; then
	echo "Info: Detected ARM architecture. Using ARM base image."
	# dockerhub doesn't have an official pytorch image for ARM, so we use NVIDIA's NGC registry
	if [ -z "$CR_URL" ] || [ "$CR_URL" = "docker.io" ]; then
		CR_URL="nvcr.io"
	fi
	MLWM_BASE_IMAGE="${CR_URL}/nvidia/pytorch:26.01-py3"
else
	echo "Info: Using x86_64 base image."
	MLWM_BASE_IMAGE="${CR_URL}/pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime"
fi

# Where the inference artifact comes from (see Containerfile): "local" uses the
# directory inference_artifact/ assembled with dev-utils/assemble_artifact.py,
# "zenodo" downloads the published inference package. The model checkpoint is
# downloaded from Zenodo during the build in both cases (unless present).
ARTIFACT_SOURCE=${ARTIFACT_SOURCE:-local}
# published ANNA inference package on Zenodo (set once the record is published)
PACKAGE_URL=${PACKAGE_URL:-}
PACKAGE_MD5=${PACKAGE_MD5:-}

if [ "$ARTIFACT_SOURCE" = "local" ]; then
	if [ ! -f inference_artifact/artifact.yaml ]; then
		echo "Error: ARTIFACT_SOURCE=local but inference_artifact/ hasn't been assembled."
		echo "Create it with dev-utils/assemble_artifact.py (see its docstring),"
		echo "or set ARTIFACT_SOURCE=zenodo."
		exit 1
	fi
	if grep -q "boundary_stats_placeholder: true" inference_artifact/artifact.yaml; then
		echo "Warning: inference_artifact/ uses PLACEHOLDER boundary statistics."
	fi
elif [ "$ARTIFACT_SOURCE" = "zenodo" ]; then
	if [ -z "$PACKAGE_URL" ] || [ -z "$PACKAGE_MD5" ]; then
		echo "Error: ARTIFACT_SOURCE=zenodo needs PACKAGE_URL and PACKAGE_MD5."
		exit 1
	fi
	# the Containerfile always copies inference_artifact/, so make sure it exists
	mkdir -p inference_artifact
else
	echo "Error: unknown ARTIFACT_SOURCE=$ARTIFACT_SOURCE (local or zenodo)"
	exit 1
fi

# Pull base image with proxy
HTTP_PROXY="$MLWM_PULL_PROXY" HTTPS_PROXY="$MLWM_PULL_PROXY" ${CONTAINER_APP} --log-level="$MLWM_LOG_LEVEL" pull "$MLWM_BASE_IMAGE"

# Build image
echo "Running ${CONTAINER_APP} build to create image $MLWM_IMAGE_NAME ..."
${CONTAINER_APP} build \
	--build-arg BASE_IMAGE="$MLWM_BASE_IMAGE" \
	--build-arg ARTIFACT_SOURCE="$ARTIFACT_SOURCE" \
	--build-arg PACKAGE_URL="$PACKAGE_URL" \
	--build-arg PACKAGE_MD5="$PACKAGE_MD5" \
	-t "$MLWM_IMAGE_NAME" \
	-f Containerfile \
	.
