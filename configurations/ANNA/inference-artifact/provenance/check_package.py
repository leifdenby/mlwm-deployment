"""
Check an inference package zip before publishing it: the zip holds the same
files as the assembled directory (byte-identical, the checkpoint only in the
directory), has its contents at the zip root, and no text file in it
mentions local or HPC paths or gefion-1 files.

Usage (from the repository root):
    uv run --project configurations/ANNA python \\
        configurations/ANNA/inference-artifact/provenance/check_package.py \\
        configurations/ANNA/inference-artifact/build/<artifact-name>.zip
"""
import argparse
import hashlib
import re
import sys
import zipfile
from pathlib import Path

LEAK_PATTERN = re.compile(r"/Users/|/home/|/private/|/var/folders/|/dcai/")
TEXT_SUFFIXES = {".yaml", ".md", ".zattrs", ".zgroup", ".zarray", ".zmetadata"}
TEXT_NAMES = {"zarr.json", ".zattrs", ".zgroup", ".zarray", ".zmetadata"}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("zip", type=Path)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="assembled directory (default: the zip path without .zip)",
    )
    args = parser.parse_args()
    root = args.artifact_dir or args.zip.with_suffix("")

    dir_files = {
        str(p.relative_to(root)): hashlib.md5(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }
    problems = []
    with zipfile.ZipFile(args.zip) as zf:
        zip_files = {
            n: hashlib.md5(zf.read(n)).hexdigest()
            for n in zf.namelist()
            if not n.endswith("/")
        }
        for name in zip_files:
            fp = Path(name)
            if fp.suffix in TEXT_SUFFIXES or fp.name in TEXT_NAMES:
                text = zf.read(name).decode(errors="replace")
                for line in text.splitlines():
                    if LEAK_PATTERN.search(line):
                        problems.append(
                            f"local path in {name}: {line.strip()}"
                        )

    only_dir = sorted(set(dir_files) - set(zip_files))
    only_zip = sorted(set(zip_files) - set(dir_files))
    differ = sorted(
        n
        for n in set(zip_files) & set(dir_files)
        if zip_files[n] != dir_files[n]
    )
    top_level = sorted({n.split("/")[0] for n in zip_files})
    print(f"zip: {len(zip_files)} files, top level: {top_level}")
    print(f"only in directory: {only_dir}")

    if only_zip or differ:
        problems.append(f"only in zip: {only_zip}, differing: {differ}")
    if [n for n in only_dir if not n.endswith(".ckpt")]:
        problems.append(f"files missing from the zip: {only_dir}")
    if [n for n in zip_files if n.endswith(".ckpt")]:
        problems.append("the zip contains a checkpoint")
    if [n for n in zip_files if "gefion-1" in n]:
        problems.append("the zip contains gefion-1 files")
    for required in ["README.md", "artifact.yaml", "configs/model.yaml"]:
        if required not in zip_files:
            problems.append(f"{required} missing from the zip")

    for problem in problems:
        print("PROBLEM:", problem)
    print("OK" if not problems else f"{len(problems)} problem(s)")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
