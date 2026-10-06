"""
Print title, authors (with affiliations), DOI and date of an arXiv paper.

Used for the creators/contributors and acknowledgements of the inference
package Zenodo record (`zenodo_draft.py`), which acknowledges all authors of
arXiv:2504.09340.

Usage (from the repository root):
    uv run --project configurations/ANNA python \\
        configurations/ANNA/inference-artifact/provenance/arxiv_meta.py \\
        2504.09340
"""
import argparse
import xml.etree.ElementTree as ET

import requests

NS = {
    "a": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("arxiv_id")
    args = parser.parse_args()

    try:  # use the OS trust store (e.g. behind a TLS-inspecting proxy)
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        pass

    r = requests.get(
        "https://export.arxiv.org/api/query",
        params=dict(id_list=args.arxiv_id),
        timeout=60,
    )
    r.raise_for_status()
    entry = ET.fromstring(r.text).find("a:entry", NS)
    print("title:", " ".join(entry.find("a:title", NS).text.split()))
    print("published:", entry.find("a:published", NS).text)
    for author in entry.findall("a:author", NS):
        aff = author.find("arxiv:affiliation", NS)
        aff = f" ({aff.text})" if aff is not None else ""
        print(f"author: {author.find('a:name', NS).text}{aff}")
    doi = entry.find("arxiv:doi", NS)
    print("doi:", doi.text if doi is not None else None)


if __name__ == "__main__":
    main()
