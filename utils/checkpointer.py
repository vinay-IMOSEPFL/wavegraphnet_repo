# utils/checkpointer.py
"""
Shared checkpoint utilities.

Saving
------
save_checkpoint(path, config, **state_dicts)
    Saves one or more model/optimizer state dicts plus a config dict.

    Example:
        save_checkpoint(
            path,
            config=vars(args),
            inv_model=inv_model.state_dict(),
            fwd_model=fwd_model.state_dict(),
        )

Loading
-------
ckpt = load_checkpoint(path)
    Returns the raw dict saved by save_checkpoint.
    Access state dicts with  ckpt["inv_model"], ckpt["config"], etc.

Naming convention (used by all main_*.py scripts)
--------------------------------------------------
checkpoints/<split>/<safe_model_name>_seed<seed>.pt

    e.g.  checkpoints/B/1D_CNN_seed42.pt
          checkpoints/B/WaveGraphNet_Coupled_seed0.pt
"""

import os
import re
import torch


def _safe_name(name: str) -> str:
    """Turn a human-readable model label into a filesystem-safe filename stem."""
    return re.sub(r"[^\w]", "_", name).strip("_")


def checkpoint_path(split: str, model_label: str, seed: int,
                    root: str = "checkpoints") -> str:
    """Return the canonical path for a checkpoint file."""
    stem = f"{_safe_name(model_label)}_seed{seed}.pt"
    return os.path.join(root, split, stem)


def save_checkpoint(path: str, config: dict, test_loss: float, **state_dicts):
    """
    Save a checkpoint.

    Parameters
    ----------
    path        : destination file (directories created automatically)
    config      : dict of hyperparameters / argparse namespace → dict
    test_loss   : final evaluation loss
    **state_dicts : any number of named state dicts, e.g.
                    model=model.state_dict()
                    inv_model=inv_model.state_dict()
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"config": config, "test_loss": test_loss}
    payload.update(state_dicts)
    torch.save(payload, path)
    print(f"  [checkpoint] saved → {path}  (test_loss={test_loss:.6f})")


def load_checkpoint(path: str) -> dict:
    """Load and return the checkpoint dict from *path*."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu")