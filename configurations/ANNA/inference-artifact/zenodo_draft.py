"""
Create (or update) a DRAFT Zenodo deposition for the ANNA inference package.

The draft gets the package zip (from `assemble_artifact.py --zip`) and the
record metadata below. It is never published by this script: review the draft
in the Zenodo web UI and publish it there (publishing is irreversible). Once
published, put the file's download URL and md5 into build_image.sh /
Containerfile (`PACKAGE_URL`, `PACKAGE_MD5`).

The Zenodo personal access token (scope `deposit:write`) is read from the
`ZENODO_TOKEN` environment variable and is not stored anywhere.

Usage:
    # review the metadata without contacting Zenodo
    python zenodo_draft.py --zip build/anna-danra-2026-10-06.zip --dry-run
    # rehearse on sandbox.zenodo.org (needs a sandbox token)
    ZENODO_TOKEN=... python zenodo_draft.py --zip ... --sandbox
    # create the draft on zenodo.org
    ZENODO_TOKEN=... python zenodo_draft.py --zip build/anna-danra-2026-10-06.zip
    # update an existing draft (replace the file, update the metadata)
    ZENODO_TOKEN=... python zenodo_draft.py --zip ... --deposition-id 1234567
"""
import argparse
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path

import yaml

ZENODO = "https://zenodo.org"
ZENODO_SANDBOX = "https://sandbox.zenodo.org"

CREATORS = [
    {
        "name": "Hintz, Kasper Stener",
        "affiliation": "Danish Meteorological Institute",
        "orcid": "0000-0002-6835-8733",
    },
    {"name": "Denby, Leif", "affiliation": "Danish Meteorological Institute"},
]
PAPER_CONTRIBUTORS = [
    {
        "name": "Adamov, Simon",
        "affiliation": "ETH Zürich",
        "orcid": "0000-0003-3599-6816",
    },
    {
        "name": "Oskarsson, Joel",
        "affiliation": "Linköping University",
        "orcid": "0000-0002-8201-0282",
    },
    {"name": "Landelius, Tomas"},
    {"name": "Christiansen, Simon"},
    {"name": "Schicker, Irene"},
    {"name": "Osuna, Carlos"},
    {"name": "Lindsten, Fredrik"},
    {"name": "Fuhrer, Oliver"},
    {"name": "Schemm, Sebastian"},
]  # fmt: skip
PAPER_TITLE = (
    "Building Machine Learning Limited Area Models: Kilometer-Scale Weather "
    "Forecasting in Realistic Settings"
)


def _md5(fp):
    md5 = hashlib.md5()
    with open(fp, "rb") as fh:
        for chunk in iter(lambda: fh.read(2**20), b""):
            md5.update(chunk)
    return md5.hexdigest()


def build_metadata(fp_zip):
    """Zenodo deposition metadata, partly read from the package itself."""
    with zipfile.ZipFile(fp_zip) as zf:
        artifact = yaml.safe_load(zf.read("artifact.yaml"))
        model = yaml.safe_load(zf.read("configs/model.yaml"))
    ckpt = model["checkpoint"]
    stats_attrs = artifact["assembled_from"]["boundary_stats_attrs"]
    version = artifact["artifact_name"].rsplit("-", 3)[-3:]
    description = f"""
<p>Inference package for running the DANRA machine-learning limited area model
(LAM) of <a href="https://arxiv.org/abs/2504.09340">Adamov et al. (2025),
"{PAPER_TITLE}"</a>
operationally from DMI's DINI forecasts (interior initial states) with DINI or
IFS forecasts on the boundary, with the
<a href="https://github.com/leifdenby/mlwm-deployment">mlwm-deployment</a>
ANNA configuration.</p>

<p><b>This package does not contain the model weights.</b> The checkpoint is the
paper's published DANRA model, <a href="https://doi.org/{ckpt['doi']}">{ckpt['doi']}</a>
(<code>danra_model.ckpt</code>, md5 <code>{ckpt['md5']}</code>), which the
deployment downloads separately.</p>

<p>Contents:</p>
<ul>
<li><code>configs/</code>: mllam-data-prep datastore configs for the DANRA interior
and the boundary (ERA5 as in training, operational IFS and DINI), neural-lam
configs, and <code>model.yaml</code> describing the model (checkpoint, graph
recipe <code>{model['graph']['name']}</code>, neural-lam arguments).
<code>ifs_7deg_model1_config.yaml</code> documents the variables, units and
grid an IFS boundary forecast must provide.</li>
<li><code>configs/era_7deg_model1_config.zarr</code>: ERA5 boundary statistics
datastore used to normalise the (ERA5, IFS or DINI) boundary forcing.</li>
<li><code>stats/</code>: training statistics of the DANRA interior datastore and
of the ERA5 boundary datastore. The ERA5 boundary statistics were recomputed
exactly from WeatherBench2 ERA5 ({stats_attrs.get('n_time_steps')} 6-hourly
steps, {stats_attrs.get('split_start')} to {stats_attrs.get('split_end')},
{stats_attrs.get('n_grid_points')} grid points), reproducing the computation of
the mllam-data-prep version used for training.</li>
<li><code>grids/</code>: the DANRA grid and static fields (from DANRA v0.5.0), and
the 18014 ERA5 boundary grid points the model was trained with.</li>
<li><code>README.md</code>: description of the package and how to use it.</li>
<li><code>artifact.yaml</code>: provenance of the package.</li>
</ul>

<p>Software: neural-lam from
<a href="https://github.com/joeloskarsson/neural-lam-dev">
joeloskarsson/neural-lam-dev</a>
(research branch) and mllam-data-prep from sadamov/mllam-data-prep
(building-ml-lams branch), as pinned in the mlwm-deployment ANNA configuration.</p>

<p>Acknowledgements: the model, the training setup and the datastore
configurations are from the work of all authors of the paper: Simon Adamov,
Joel Oskarsson, Leif Denby, Tomas Landelius, Kasper Hintz, Simon Christiansen,
Irene Schicker, Carlos Osuna, Fredrik Lindsten, Oliver Fuhrer and Sebastian
Schemm.</p>
"""
    return {
        "upload_type": "dataset",
        "title": (
            "ANNA inference package for the DANRA ML LAM model "
            "(DINI/IFS-driven forecasts)"
        ),
        "creators": CREATORS,
        "contributors": [
            dict(c, type="Researcher") for c in PAPER_CONTRIBUTORS
        ],
        "description": " ".join(description.split()),
        "access_right": "open",
        "license": "cc-by-4.0",
        "version": "-".join(version),
        "keywords": [
            "machine learning weather prediction",
            "limited area model",
            "neural-lam",
            "DANRA",
            "DINI",
            "IFS",
            "ERA5",
        ],
        "related_identifiers": [
            {
                "identifier": ckpt["doi"],
                "relation": "requires",
                "resource_type": "other",
                "scheme": "doi",
            },
            {
                "identifier": "arXiv:2504.09340",
                "relation": "isSupplementTo",
                "resource_type": "publication-preprint",
                "scheme": "arxiv",
            },
            {
                "identifier": "https://github.com/joeloskarsson/neural-lam-dev",
                "relation": "requires",
                "resource_type": "software",
                "scheme": "url",
            },
            {
                "identifier": "https://github.com/leifdenby/mlwm-deployment",
                "relation": "isCompiledBy",
                "resource_type": "software",
                "scheme": "url",
            },
        ],
        "prereserve_doi": True,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--zip", required=True, type=Path)
    parser.add_argument("--sandbox", action="store_true")
    parser.add_argument("--deposition-id", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    metadata = build_metadata(args.zip)
    md5 = _md5(args.zip)
    print(f"{args.zip.name}: {args.zip.stat().st_size} bytes, md5 {md5}")
    if args.dry_run:
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    import requests

    try:  # use the OS trust store (e.g. behind a TLS-inspecting proxy)
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        pass

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        sys.exit("ZENODO_TOKEN is not set")
    base = ZENODO_SANDBOX if args.sandbox else ZENODO
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {token}"

    if args.deposition_id is None:
        r = session.post(
            f"{base}/api/deposit/depositions", json={}, timeout=60
        )
    else:
        r = session.get(
            f"{base}/api/deposit/depositions/{args.deposition_id}", timeout=60
        )
    r.raise_for_status()
    deposition = r.json()
    if deposition.get("submitted"):
        sys.exit(f"deposition {deposition['id']} is already published")
    dep_id, bucket = deposition["id"], deposition["links"]["bucket"]

    # remove an earlier upload of the same file from the draft
    for f in deposition.get("files", []):
        if f["filename"] == args.zip.name:
            session.delete(f["links"]["self"], timeout=60).raise_for_status()

    print(f"uploading {args.zip.name} to draft {dep_id} ...")
    with open(args.zip, "rb") as fh:
        r = session.put(f"{bucket}/{args.zip.name}", data=fh, timeout=3600)
    r.raise_for_status()
    uploaded_md5 = r.json()["checksum"].removeprefix("md5:")
    if uploaded_md5 != md5:
        sys.exit(f"uploaded file md5 {uploaded_md5} != local {md5}")

    r = session.put(
        f"{base}/api/deposit/depositions/{dep_id}",
        json={"metadata": metadata},
        timeout=60,
    )
    r.raise_for_status()
    deposition = r.json()
    doi = deposition["metadata"].get("prereserve_doi", {}).get("doi")
    print(f"draft {dep_id} ready for review: {deposition['links']['html']}")
    print(f"reserved DOI: {doi}")
    print(f"file md5: {md5} (PACKAGE_MD5)")
    print(
        f"after publishing, PACKAGE_URL={base}/records/{dep_id}/files/"
        f"{args.zip.name}?download=1"
    )
    print("NOT published: review and publish it in the Zenodo web UI")


if __name__ == "__main__":
    main()
