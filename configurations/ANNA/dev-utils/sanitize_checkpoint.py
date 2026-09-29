"""
Remove the pickled training boundary datastore from a neural-lam checkpoint.

The gefion-1 checkpoint stores the boundary `MDPDatastore` used during training
in its `hyper_parameters["datastore_boundary"]`. That object holds lazy (dask)
arrays pointing at the training data on Gefion via zarr v2 `zarr.core.Array`
objects, so the checkpoint can't be unpickled with zarr v3 (and even with zarr
v2 the training data paths don't exist). The object is not needed for
inference, as `train_model --eval` passes the inference datastores to
`load_from_checkpoint`, so it is dropped here. Model weights, `args` and
`config` are kept unchanged.

Usage:
    python sanitize_checkpoint.py <checkpoint_in> <checkpoint_out>
"""
import argparse
import pickle
import types

import torch

KEYS_TO_DROP = ["datastore_boundary"]


class _Unloadable:
    """Stand-in for classes that can't be imported (e.g. zarr v2 classes)."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            return type(f"{module}.{name}", (_Unloadable,), {})


_pickle_module = types.ModuleType("sanitize_pickle")
_pickle_module.__dict__.update(pickle.__dict__)
_pickle_module.Unpickler = _Unpickler


def _contains_unloadable(obj, _seen=None):
    _seen = set() if _seen is None else _seen
    if id(obj) in _seen:
        return False
    _seen.add(id(obj))
    if isinstance(obj, _Unloadable):
        return True
    if isinstance(obj, dict):
        return any(_contains_unloadable(v, _seen) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(_contains_unloadable(v, _seen) for v in obj)
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        return _contains_unloadable(vars(obj), _seen)
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint_in")
    parser.add_argument("checkpoint_out")
    args = parser.parse_args()

    ckpt = torch.load(
        args.checkpoint_in,
        map_location="cpu",
        weights_only=False,
        pickle_module=_pickle_module,
    )
    hparams = ckpt["hyper_parameters"]
    for key in KEYS_TO_DROP:
        if key in hparams:
            print(f"Dropping hyper_parameters[{key!r}]")
            del hparams[key]

    for key, value in hparams.items():
        if _contains_unloadable(value):
            raise ValueError(
                f"hyper_parameters[{key!r}] still contains objects that could "
                "not be unpickled, refusing to write a checkpoint with stubs"
            )

    torch.save(ckpt, args.checkpoint_out)
    print(
        f"Wrote {args.checkpoint_out} (kept hyper_parameters: {list(hparams)})"
    )


if __name__ == "__main__":
    main()
