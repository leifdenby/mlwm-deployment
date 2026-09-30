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
# "s3" downloads it from S3
ARTIFACT_SOURCE=${ARTIFACT_SOURCE:-local}

if [ "$ARTIFACT_SOURCE" = "local" ]; then
	if [ ! -f inference_artifact/artifact.yaml ]; then
		echo "Error: ARTIFACT_SOURCE=local but inference_artifact/ hasn't been assembled."
		echo "Create it with dev-utils/assemble_artifact.py (see its docstring),"
		echo "or set ARTIFACT_SOURCE=s3."
		exit 1
	fi
	if grep -q "boundary_stats_placeholder: true" inference_artifact/artifact.yaml; then
		echo "Warning: inference_artifact/ uses PLACEHOLDER boundary statistics."
	fi
elif [ "$ARTIFACT_SOURCE" = "s3" ]; then
	# the Containerfile always copies inference_artifact/, so make sure it exists
	mkdir -p inference_artifact
	# Check AWS credentials, S3 access is needed
	if [ -z "$AWS_ACCESS_KEY_ID" ]; then
		echo "Error: AWS_ACCESS_KEY_ID is not set. Please set it before running this script."
		exit 1
	fi
	if [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
		echo "Error: AWS_SECRET_ACCESS_KEY is not set. Please set it before running this script."
		exit 1
	fi
	if [ -z "$AWS_DEFAULT_REGION" ]; then
		echo "Error: AWS_DEFAULT_REGION is not set. We set it automatically to eu-central-1."
		AWS_DEFAULT_REGION="eu-central-1"
	fi
else
	echo "Error: unknown ARTIFACT_SOURCE=$ARTIFACT_SOURCE (local or s3)"
	exit 1
fi

# Pull base image with proxy
HTTP_PROXY="$MLWM_PULL_PROXY" HTTPS_PROXY="$MLWM_PULL_PROXY" ${CONTAINER_APP} --log-level="$MLWM_LOG_LEVEL" pull "$MLWM_BASE_IMAGE"

# Build image with AWS credentials as build arguments
echo "Running ${CONTAINER_APP} build to create image $MLWM_IMAGE_NAME ..."
${CONTAINER_APP} build \
	--build-arg BASE_IMAGE="$MLWM_BASE_IMAGE" \
	--build-arg ARTIFACT_SOURCE="$ARTIFACT_SOURCE" \
	--build-arg AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
	--build-arg AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
    --build-arg AWS_DEFAULT_REGION="$AWS_DEFAULT_REGION" \
	-t "$MLWM_IMAGE_NAME" \
	-f Containerfile \
	.
