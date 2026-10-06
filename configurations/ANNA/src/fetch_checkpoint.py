"""
Download the model checkpoint described in an inference artifact's
`configs/model.yaml` into the artifact directory, and check its md5.

Does nothing if the checkpoint is already there with the right md5.

Usage:
    python fetch_checkpoint.py <artifact_dir>
"""
import argparse
import hashlib
import urllib.request
from pathlib import Path

import yaml
from loguru import logger


def _md5(fp):
    md5 = hashlib.md5()
    with open(fp, "rb") as fh:
        for chunk in iter(lambda: fh.read(2**20), b""):
            md5.update(chunk)
    return md5.hexdigest()


def fetch_checkpoint(artifact_dir):
    artifact_dir = Path(artifact_dir)
    try:  # use the OS trust store (e.g. behind a TLS-inspecting proxy)
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        pass

    model = yaml.safe_load(
        (artifact_dir / "configs" / "model.yaml").read_text()
    )
    ckpt = model["checkpoint"]
    fp = artifact_dir / ckpt["artifact_path"]
    if fp.exists() and _md5(fp) == ckpt["md5"]:
        logger.info(f"checkpoint already present and verified: {fp}")
        return fp

    fp_tmp = fp.with_name(fp.name + ".tmp")
    logger.info(f"downloading {ckpt['url']} to {fp}")
    with urllib.request.urlopen(ckpt["url"]) as response, open(
        fp_tmp, "wb"
    ) as fh:
        for chunk in iter(lambda: response.read(2**20), b""):
            fh.write(chunk)
    md5 = _md5(fp_tmp)
    if md5 != ckpt["md5"]:
        fp_tmp.unlink()
        raise ValueError(
            f"md5 of downloaded checkpoint {md5} != {ckpt['md5']}"
        )
    fp_tmp.replace(fp)
    logger.info(f"checkpoint downloaded and verified: {fp}")
    return fp


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("artifact_dir", type=Path)
    args = parser.parse_args()
    fetch_checkpoint(args.artifact_dir)


if __name__ == "__main__":
    main()
