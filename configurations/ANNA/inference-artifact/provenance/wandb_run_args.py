"""
Print the command line, git commit and training config of a wandb run.

Used to recover how the paper's DANRA model was trained (run `hfzfhiha`, see
`configs/model.yaml`): the `train_model` arguments, the neural-lam-dev commit
and the datastore config. Needs wandb access to the `jo-research-team`
entity.

Usage (from the repository root):
    uv run --project configurations/ANNA --with wandb python \\
        configurations/ANNA/inference-artifact/provenance/wandb_run_args.py \\
        jo-research-team/neural_lam/hfzfhiha
"""
import argparse
import json
import tempfile

import wandb


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run", help="entity/project/run_id")
    args = parser.parse_args()

    run = wandb.Api().run(args.run)
    print("name:", run.name, "created:", run.created_at)
    with tempfile.TemporaryDirectory() as tmpdir:
        with run.file("wandb-metadata.json").download(root=tmpdir) as fh:
            meta = json.load(fh)
    print("git:", meta.get("git"))
    print("args:", " ".join(meta.get("args", [])))
    training = dict(run.config).get("training", {})
    for k in sorted(training):
        print(f"  training.{k}: {training[k]}")


if __name__ == "__main__":
    main()
