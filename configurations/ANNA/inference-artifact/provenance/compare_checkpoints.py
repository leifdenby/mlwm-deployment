"""
Compare two neural-lam checkpoints: training progress, stored arguments and
weights.

This is how the paper's DANRA checkpoint (Zenodo 15131838) was found to be a
different model from gefion-1 (different tensors and graph levels, no stored
arguments), see README.md. Each checkpoint is given as a `.ckpt` file, an
inference artifact zip with `checkpoint.pkl` in it (e.g. gefion-1.zip) or a
URL; everything is loaded in memory.

Usage (from the repository root):
    uv run --project configurations/ANNA python \\
        configurations/ANNA/inference-artifact/provenance/compare_checkpoints.py \\
        gefion-1.zip \\
        "https://zenodo.org/records/15131838/files/danra_model.ckpt?download=1"
"""
import argparse
import io
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from sanitize_checkpoint import load_checkpoint  # noqa: E402


def _read(source):
    if source.startswith(("http://", "https://")):
        import requests

        try:  # use the OS trust store (e.g. behind a TLS-inspecting proxy)
            import truststore

            truststore.inject_into_ssl()
        except ImportError:
            pass
        print(f"downloading {source} into memory ...", flush=True)
        r = requests.get(source, timeout=600)
        r.raise_for_status()
        return io.BytesIO(r.content)
    if source.endswith(".zip"):
        with zipfile.ZipFile(source) as zf:
            return io.BytesIO(zf.read("checkpoint.pkl"))
    return source


def _summarise(name, ckpt):
    print(
        f"\n== {name}: epoch {ckpt.get('epoch')}, "
        f"global_step {ckpt.get('global_step')}, "
        f"lightning {ckpt.get('pytorch-lightning_version')}"
    )
    print("   top-level keys:", list(ckpt))
    print("   hyper_parameters keys:", list(ckpt.get("hyper_parameters", {})))
    for key, value in ckpt.get("callbacks", {}).items():
        if "ModelCheckpoint" in str(key) and isinstance(value, dict):
            for k in ["best_model_path", "dirpath", "best_model_score"]:
                if k in value:
                    print(f"   {k}: {value[k]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint_a")
    parser.add_argument("checkpoint_b")
    args = parser.parse_args()

    a = load_checkpoint(_read(args.checkpoint_a))
    b = load_checkpoint(_read(args.checkpoint_b))
    _summarise(args.checkpoint_a, a)
    _summarise(args.checkpoint_b, b)

    args_a = a.get("hyper_parameters", {}).get("args")
    args_b = b.get("hyper_parameters", {}).get("args")
    if args_a is not None and args_b is not None:
        va, vb = vars(args_a), vars(args_b)
        print("\nstored train_model argument differences (a, b):")
        for k in sorted(set(va) | set(vb)):
            if va.get(k) != vb.get(k):
                print(f"  {k}: {va.get(k)!r}, {vb.get(k)!r}")
    else:
        print("\nat least one checkpoint has no stored train_model arguments")

    sa, sb = a["state_dict"], b["state_dict"]
    print(
        f"\nstate_dict: {len(sa)} vs {len(sb)} tensors, "
        f"same keys: {set(sa) == set(sb)}"
    )
    print("only in a:", sorted(set(sa) - set(sb))[:10])
    print("only in b:", sorted(set(sb) - set(sa))[:10])
    common = [k for k in sa if k in sb and sa[k].shape == sb[k].shape]
    print(
        "shape differences:",
        [k for k in sa if k in sb and sa[k].shape != sb[k].shape][:10],
    )
    rel = sorted(
        (float((sa[k] - sb[k]).norm() / (sa[k].norm() + 1e-12)), k)
        for k in common
        if sa[k].is_floating_point()
    )
    if rel:
        print(
            f"relative weight differences of common tensors: "
            f"median {rel[len(rel) // 2][0]:.3g}, max {rel[-1][0]:.3g} "
            f"({rel[-1][1]}), identical: {all(d == 0 for d, _ in rel)}"
        )


if __name__ == "__main__":
    main()
