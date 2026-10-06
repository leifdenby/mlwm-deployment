"""
Print the metadata of a published Zenodo record (read-only, no token needed).

Used to check the paper's DANRA checkpoint record (15131838: licence,
creators, file md5) that `configs/model.yaml` and the inference package
record (`zenodo_draft.py`) refer to.

Usage (from the repository root):
    uv run --project configurations/ANNA python \\
        configurations/ANNA/inference-artifact/provenance/zenodo_record.py \\
        15131838
"""
import argparse
import json
import re

import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("record_id")
    args = parser.parse_args()

    try:  # use the OS trust store (e.g. behind a TLS-inspecting proxy)
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        pass

    rec = requests.get(
        f"https://zenodo.org/api/records/{args.record_id}", timeout=60
    ).json()
    md = rec["metadata"]
    print("title:", md.get("title"))
    print("doi:", rec.get("doi"), "| concept doi:", rec.get("conceptdoi"))
    print(
        "published:",
        md.get("publication_date"),
        "| version:",
        md.get("version"),
        "| resource type:",
        md.get("resource_type"),
    )
    print("license:", md.get("license"))
    print("access:", md.get("access_right"))
    for key in ["creators", "contributors", "related_identifiers"]:
        print(f"{key}:", json.dumps(md.get(key), indent=1))
    print("keywords:", md.get("keywords"))
    description = re.sub("<[^>]+>", " ", md.get("description", ""))
    print("description:", re.sub(r"\s+", " ", description))
    for f in rec["files"]:
        print(f"file: {f['key']} {f['size']} bytes checksum {f['checksum']}")


if __name__ == "__main__":
    main()
