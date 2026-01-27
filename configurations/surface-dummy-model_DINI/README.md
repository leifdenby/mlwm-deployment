# surface-dummy-model_DINI

The model configuration in this directory is a dummy model that was trained on
surface variables from DANRA, only 10 days of data and only trained 10
epochs. It is intended only as a demonstration of the inference pipeline and is
expected to give very poor results.

## Building image and running inference

Currently building the image and running inference is only supported on the "superjuice" machine (`27sj894.dmi.dk`).

### Building the image

To build the image on "superjuice" (`27sj894.dmi.dk`) we need to set the AWS tokens to read the inference artifact and also use the local http proxy for pulling the base image:

```bash
export AWS_SECRET_ACCESS_KEY=<secret-key-to-read-inference-artifact>
export AWS_ACCESS_KEY_ID=<access-key-to-read-inference-artifact>
export MLWM_PULL_PROXY=http://squid1.dmi.dk:3128
```

Then build the image with:

```bash
./build_image.sh
```

### Running inference

On "superjuice" (`27sj894.dmi.dk`), run inference for a given analysis time (e.g. `2019-02-04T12:00`) and forecast duration (e.g. `PT18H`) using DINI initial conditions (read from AWS S3) with:

```bash
./run_inference_container.sh 2019-02-04T12:00 PT18H
```

Currently this script uses a workaround to get GPU access with rootless Podman. This is required because the necessary system-level NVIDIA Container Toolkit integration is not available on this system. This means that the standard Podman/Docker flag:

  --gpus all

does not work out of the box, even though the host has a functioning NVIDIA driver and GPUs.

WHY THIS IS NECESSARY

Normally, GPU support in containers relies on the NVIDIA Container Toolkit, which at runtime:

- exposes /dev/nvidia* device nodes to the container
- bind-mounts the host NVIDIA driver libraries (most importantly libcuda.so.1)
- injects utilities such as nvidia-smi

In a rootless Podman setup without system-level NVIDIA integration:

- --gpus all is a no-op
- libcuda.so.1 is not available inside the container
- CUDA frameworks (PyTorch, Lightning, etc.) report that no GPU is available

WORKING COMMAND (ROOTLESS, NO SUDO)

```bash
podman run --rm \
  --device /dev/nvidia0 \
  --device /dev/nvidiactl \
  --device /dev/nvidia-uvm \
  --device /dev/nvidia-uvm-tools \
  --device /dev/nvidia-modeset \
  --shm-size=32g \
  -v /lib/x86_64-linux-gnu/libcuda.so.1:/lib/x86_64-linux-gnu/libcuda.so.1:ro \
  -v /lib/x86_64-linux-gnu/libnvidia-ml.so.1:/lib/x86_64-linux-gnu/libnvidia-ml.so.1:ro \
  -v /lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1:/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1:ro \
  -v ./inference_workdir/:/workspace/inference_workdir/ \
  localhost/surface-dummy-model_dini:latest
```

With this setup, CUDA becomes available inside the container.

WHAT IS NEEDED TO USE `--gpus all` INSTEAD (RECOMMENDED)

To enable the standard workflow:

```bash
podman run --gpus all ...
```

the following needs to be provided system-wide by IT:

1. Install NVIDIA Container Toolkit on the host
2. Enable Container Device Interface (CDI) or OCI hooks for Podman
3. Generate the NVIDIA CDI specification using:
     nvidia-ctk cdi generate
4. Ensure Podman is configured to consume CDI devices

Once enabled:
- GPU devices and driver libraries are injected automatically
- nvidia-smi works inside containers
- No manual --device or library mounts are required
- --gpus all works as expected

## Upstream package change requirements

Relative to the `main` branch on both github.com/mllam/mllam-data-prep and
github.com/mllam/neural-lam and number of pieces of functionality are currently
required to run this configuration:

**mllam-data-prep**:

using branch `feat/inference-cli-args` on
https://github.com/leifdenby/mllam-data-prep@feat/inference-cli-args, which adds:

- functionality to invert datasets created by `mllam-data-prep` back to the
  structure of the input datasets that we were used. In the current
  configuration that is used to restructure the forecast zarr dataset that
  `neural-lam` outputs during inference back to the structure of the input
  forecast dataset.

  - also in seperate branch and PR: https://github.com/leifdenby/mllam-data-prep/tree/feat/inverse-ops

- use of cf-compliant encoding of `xarray/pandas` `MultiIndex` coordinates to
  store stacked coordinates. This is required since we `MultiIndex` coordinates
  can't natively be stored in zarr/netcdf files, but fortunately `cf_xarray`
  have implemented the cf-compliant way of handling this (see
  https://cf-xarray.readthedocs.io/en/latest/coding.html)

  - needs its own branch and PR

- support for supplying statistics from the training dataset during creation of
  the inference dataset, so that the inference dataset can be normalised in the
  same way as the training dataset.

  - needs its own branch and PR

- support for selecting only a single value from a variable/coordinate in the
  configuration. This is used to select only a single analysis time during
  creation of the inference dataset.

  - needs its own branch and PR


**neural-lam**:

using branch `dev/first-inference-image` on
https://github.com/leifdenby/neural-lam/tree/dev/first-inference-image, which
adds:

- support for decoding cf-compliant `MultiIndex` encoded coordinates when reading
  datasets produced with mllam-data-prep.

  - this needs its own branch and PR, and needs to be implemented so datasets
    made with previous versions of `mllam-data-prep` are still usable in `neural-lam`

- support for writing output from inference (i.e. `--eval` mode) to a zarr
  dataset. Needs to be merged after the multiindex decoding above.

  - also in seperate branch and PR: https://github.com/leifdenby/neural-lam/tree/feat/write-to-zarr

- support for using forecast data in in mllam-data-prep datastore (`MDPDatastore`)

  - needs its own branch and PR

- make logging of validation steps optional in the training CLI (i.e. `--eval` mode)

  - needs its own branch and PR

- `torch >= 2.6.0` defaults to `weights_only=True` when loading checkpoints
