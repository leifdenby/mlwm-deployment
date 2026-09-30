ARG BASE_IMAGE="pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime"
FROM ${BASE_IMAGE}

WORKDIR /workspace

COPY pyproject.toml .
COPY *.yaml ./
COPY entry.sh ./
COPY src/ ./src

# Where the inference artifact comes from:
# - "local": the artifact directory `inference_artifact/` in the build context,
#   assembled with `dev-utils/assemble_artifact.py` (gefion-1 completed with
#   the boundary datastore configs and statistics, see INFERENCE_PLAN.md)
# - "s3": download ${DEFAULT_ARTIFACT} from S3 (needs AWS credentials as build
#   args). NB: gefion-1.zip on S3 lacks the boundary datastore, so this only
#   works once a completed artifact has been uploaded
ARG ARTIFACT_SOURCE="local"
ARG DEFAULT_ARTIFACT="s3://mlwm-artifacts/inference-artifacts/gefion-1.zip"
ENV MLWM_INFERENCE_ARTIFACT=${DEFAULT_ARTIFACT}

RUN apt-get update && apt-get install -y python3 python3-pip unzip git
RUN pip3 install awscli

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_DEFAULT_REGION
ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
ENV AWS_DEFAULT_REGION=$AWS_DEFAULT_REGION

# `inference_artifact/` must exist in the build context (build_image.sh
# creates an empty one when ARTIFACT_SOURCE=s3)
COPY inference_artifact/ /workspace/inference_artifact_local/
RUN if [ "$ARTIFACT_SOURCE" = "local" ] ; then \
        mv /workspace/inference_artifact_local /workspace/inference_artifact ; \
    elif [ "$ARTIFACT_SOURCE" = "s3" ] ; then \
        rm -rf /workspace/inference_artifact_local && \
        aws s3 cp $MLWM_INFERENCE_ARTIFACT ./inference_artifact.zip && \
        mkdir -p /workspace/inference_artifact && \
        unzip inference_artifact.zip -d /workspace/inference_artifact && \
        rm inference_artifact.zip ; \
    else \
        echo "unknown ARTIFACT_SOURCE=$ARTIFACT_SOURCE (local or s3)" && exit 1 ; \
    fi
# List files in the inference artifact directory for verification
RUN ls -la /workspace/inference_artifact

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:$PATH"

# export current torch version to constraints file
RUN python -c 'import torch; print(f"torch=={torch.__version__}")' > constraints.txt
# install with constraints
RUN uv pip install --break-system-packages --system --constraints constraints.txt .
# check that we can print out neural-lam version
RUN python -c 'import neural_lam; print(f"neural-lam=={neural_lam.__version__}")'

# inside the container we have installed directly into system python, so we
# wont use uv here (otherwise uv tries to create a new virtual environment)
ENV USE_UV="false"
# Set the default command to run when the container starts
ENTRYPOINT ["./entry.sh"]
# No arguments passed by default
CMD [""]
