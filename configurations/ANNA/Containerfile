ARG BASE_IMAGE="pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime"
FROM ${BASE_IMAGE}

WORKDIR /workspace

RUN apt-get update && apt-get install -y curl unzip git

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

COPY pyproject.toml .
COPY *.yaml ./
COPY entry.sh ./
COPY src/ ./src

# export current torch version to constraints file
RUN python -c 'import torch; print(f"torch=={torch.__version__}")' > constraints.txt
# install with constraints
RUN uv pip install --break-system-packages --system --constraints constraints.txt .
# check that we can print out neural-lam version
RUN python -c 'import neural_lam; print(f"neural-lam=={neural_lam.__version__}")'

# The inference artifact (configs, statistics, grids, model description, see
# inference-artifact/README.md) comes from:
# - "local": the assembled package directory ${LOCAL_ARTIFACT_DIR} in the
#   build context
# - "zenodo": the published inference package ${PACKAGE_URL}, checked against
#   ${PACKAGE_MD5}
# The model checkpoint is then downloaded (md5-checked) from Zenodo as given in
# inference_artifact/configs/model.yaml, unless it's already there.
ARG ARTIFACT_SOURCE="local"
ARG PACKAGE_URL=""
ARG PACKAGE_MD5=""
ARG LOCAL_ARTIFACT_DIR="inference-artifact/build/anna-local"
# ${LOCAL_ARTIFACT_DIR} must exist in the build context (build_image.sh
# creates an empty one when ARTIFACT_SOURCE=zenodo)
COPY ${LOCAL_ARTIFACT_DIR}/ /workspace/inference_artifact_local/
RUN if [ "$ARTIFACT_SOURCE" = "local" ] ; then \
        mv /workspace/inference_artifact_local /workspace/inference_artifact ; \
    elif [ "$ARTIFACT_SOURCE" = "zenodo" ] ; then \
        rm -rf /workspace/inference_artifact_local && \
        test -n "$PACKAGE_URL" && test -n "$PACKAGE_MD5" && \
        curl -fL -o inference_artifact.zip "$PACKAGE_URL" && \
        echo "$PACKAGE_MD5  inference_artifact.zip" | md5sum -c - && \
        mkdir -p /workspace/inference_artifact && \
        unzip inference_artifact.zip -d /workspace/inference_artifact && \
        rm inference_artifact.zip ; \
    else \
        echo "unknown ARTIFACT_SOURCE=$ARTIFACT_SOURCE (local or zenodo)" && exit 1 ; \
    fi
RUN python src/fetch_checkpoint.py /workspace/inference_artifact
# List files in the inference artifact directory for verification
RUN ls -la /workspace/inference_artifact

# inside the container we have installed directly into system python, so we
# wont use uv here (otherwise uv tries to create a new virtual environment)
ENV USE_UV="false"
# Set the default command to run when the container starts
ENTRYPOINT ["./entry.sh"]
# No arguments passed by default
CMD [""]
