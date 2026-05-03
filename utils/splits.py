# utils/splits.py
"""
Spatial train / val / test splits for the OGW-1 SHM Plate benchmark.

Split A  (paper split)
  Test  : D4, D21–D25          — spatially held-out (right edge + bottom)
  Val   : D9, D13, D27         — spread across plate interior (held from train)
  Train : remaining damaged + baseline

Split B  (harder outer-ring split)
  Test  : D1–D4, D21–D24       — both outer corners, fully outside hull
  Val   : D7, D15, D25         — spread across remaining interior (held from train)
  Train : remaining damaged + baseline

Validation damage zones are chosen to:
  1. Come from the training pool (never overlap with test).
  2. Be spatially spread (top, middle, bottom of plate interior).
  3. Each "zone" withholds ALL repetitions of that damage label.
"""

import random


# ── Spatial split definitions ─────────────────────────────────────────────────
_SPLITS = {
    "A": {
        "test" : ["D4",  "D21", "D22", "D23", "D24", "D25"],
        # D9 (center-left), D13 (center-right), D27 (bottom-center)
        "val"  : ["D9",  "D13", "D27"],
    },
    "B": {
        "test" : ["D1", "D2", "D3", "D4", "D21", "D22", "D23", "D24"],
        # D7 (upper-center), D15 (center-right), D25 (bottom-center)
        "val"  : ["D7",  "D15", "D25"],
    },
}


def get_train_val_test_ids(
    split_name: str,
    all_sample_ids: list,
    baseline_val_ratio: float = 0.10,
    baseline_test_ratio: float = 0.10,
    seed: int = 42,
) -> tuple[list, list, list]:
    """
    Returns (train_ids, val_ids, test_ids).

    Damaged samples are assigned deterministically by label.
    Baseline samples are split randomly by ratio (reproducible via seed).

    Parameters
    ----------
    split_name        : "A" or "B"
    all_sample_ids    : full list of sample keys from data_map
    baseline_val_ratio : fraction of baseline samples used for validation
    baseline_test_ratio: fraction of baseline samples used for testing
    seed              : RNG seed for baseline shuffle only
    """
    key = split_name.upper()
    if key not in _SPLITS:
        raise ValueError(f"Unknown split '{split_name}'. Choose A or B.")

    cfg = _SPLITS[key]
    test_labels = set(cfg["test"])
    val_labels  = set(cfg["val"])

    # ── Separate by type ──────────────────────────────────────────────────────
    damaged_ids  = [s for s in all_sample_ids if not s.startswith("baseline")]
    baseline_ids = [s for s in all_sample_ids if s.startswith("baseline")]

    # ── Assign damaged samples by label ───────────────────────────────────────
    dmg_train, dmg_val, dmg_test = [], [], []
    for sid in damaged_ids:
        label = sid.split("_")[0]          # e.g. "D9" from "D9_100kHz"
        if label in test_labels:
            dmg_test.append(sid)
        elif label in val_labels:
            dmg_val.append(sid)
        else:
            dmg_train.append(sid)

    # ── Split baseline randomly ───────────────────────────────────────────────
    rng = random.Random(seed)
    bl = list(baseline_ids)
    rng.shuffle(bl)
    n   = len(bl)
    n_test = max(1, int(n * baseline_test_ratio))
    n_val  = max(1, int(n * baseline_val_ratio))
    bl_test  = bl[:n_test]
    bl_val   = bl[n_test : n_test + n_val]
    bl_train = bl[n_test + n_val :]

    train_ids = dmg_train + bl_train
    val_ids   = dmg_val   + bl_val
    test_ids  = dmg_test  + bl_test

    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    rng.shuffle(test_ids)

    _print_split_summary(split_name, train_ids, val_ids, test_ids,
                         val_labels, test_labels)
    return train_ids, val_ids, test_ids


def get_train_test_ids(split_name: str, all_sample_ids: list, **kwargs):
    """Legacy wrapper — returns (train, test) for backward compatibility."""
    train, val, test = get_train_val_test_ids(split_name, all_sample_ids, **kwargs)
    return train + val, test    # merge train+val for scripts that don't use val


def _print_split_summary(split, train_ids, val_ids, test_ids, val_labels, test_labels):
    n_dmg_tr  = sum(1 for s in train_ids if not s.startswith("baseline"))
    n_dmg_val = sum(1 for s in val_ids   if not s.startswith("baseline"))
    n_dmg_te  = sum(1 for s in test_ids  if not s.startswith("baseline"))
    n_bl_tr   = len(train_ids) - n_dmg_tr
    n_bl_val  = len(val_ids)   - n_dmg_val
    n_bl_te   = len(test_ids)  - n_dmg_te
    print(f"\n  Split {split} │ "
          f"Train: {n_dmg_tr}D+{n_bl_tr}B={len(train_ids)} │ "
          f"Val ({sorted(val_labels)}): {n_dmg_val}D+{n_bl_val}B={len(val_ids)} │ "
          f"Test ({sorted(test_labels)}): {n_dmg_te}D+{n_bl_te}B={len(test_ids)}",
          flush=True)