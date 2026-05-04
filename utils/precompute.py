# utils/precompute.py
"""
All statistics derived from the training split.
Mirrors the notebook pipeline exactly:
  Step 0: Baseline subtraction  (average_raw_baseline from training baselines)
  Step 1: Time-domain normalization  (diff_mean, diff_std from training diffs)
  Step 2: Frequency-domain amplitude stats  (per-pair-per-freq amp_means, amp_stds)
  Step 3: Average baseline energy profile  [66, 256]
  Step 4: Global max delta-E  (scalar for normalizing fwd target)
"""

import itertools
import numpy as np
import scipy.fft
import torch
from tqdm import tqdm

# ── Physical constants ────────────────────────────────────────────────────────
LOOKBACK_POINTS      = 13_108
SAMPLING_RATE        = 10_000_000
MIN_FREQ_HZ          = 69_400
MAX_FREQ_HZ          = 128_000
N_ATTENTION_FREQS    = 256
NUM_TRANSDUCERS      = 12
NUM_DATA_COLUMNS     = 66

# ── Fixed FFT bin indices (computed once at import) ───────────────────────────
def _compute_fixed_fft_bin_indices():
    freqs   = scipy.fft.rfftfreq(LOOKBACK_POINTS, d=1.0 / SAMPLING_RATE)
    targets = np.linspace(MIN_FREQ_HZ, MAX_FREQ_HZ, N_ATTENTION_FREQS)
    return np.array([np.argmin(np.abs(freqs - f)) for f in targets], dtype=np.int64)

FIXED_FFT_BIN_INDICES = _compute_fixed_fft_bin_indices()


# ── Step 0 + 1: Time-domain preprocessing ────────────────────────────────────
def build_normalized_differential_data(
    raw_data_map: dict,
    train_ids: list,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Exactly mirrors notebooks Cell 9/10/11.

    Returns
    -------
    normalized_differential_data : dict  (same keys as raw_data_map)
    average_raw_baseline          : np.ndarray  [T, 66]
    diff_mean                     : np.ndarray  [1, 66]
    diff_std                      : np.ndarray  [1, 66]
    """
    baseline_train_ids = [s for s in train_ids if "baseline" in s]

    # Step 0 – average baseline from training baselines only
    min_len  = min(raw_data_map[s].shape[0] for s in baseline_train_ids)
    stacked  = np.stack([raw_data_map[s][:min_len] for s in baseline_train_ids], axis=0)
    average_raw_baseline = stacked.mean(axis=0).astype(np.float32)

    # Step 0b – subtract baseline from every sample
    differential_data = {}
    for sid, sig in raw_data_map.items():
        L = min(sig.shape[0], average_raw_baseline.shape[0])
        differential_data[sid] = sig[:L] - average_raw_baseline[:L]

    # Step 1 – z-normalize by training-set statistics
    all_train_diff = np.concatenate([differential_data[s] for s in train_ids], axis=0)
    diff_mean = all_train_diff.mean(axis=0, keepdims=True).astype(np.float32)
    diff_std  = all_train_diff.std(axis=0,  keepdims=True).astype(np.float32)
    diff_std[diff_std == 0] = 1e-6

    normalized_differential_data = {
        sid: (diff_signal - diff_mean) / diff_std
        for sid, diff_signal in differential_data.items()
    }

    return normalized_differential_data, average_raw_baseline, diff_mean, diff_std


# ── Step 2: Per-(pair,freq) amplitude statistics ──────────────────────────────
def compute_amp_stats(
    normalized_data_map: dict,
    train_ids: list,
    fixed_fft_bin_indices: np.ndarray = FIXED_FFT_BIN_INDICES,
    lookback: int = LOOKBACK_POINTS,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns amp_means, amp_stds — both shape [66, 256]."""
    n_pairs = NUM_DATA_COLUMNS
    n_freqs = len(fixed_fft_bin_indices)
    amps_list = [[[] for _ in range(n_freqs)] for _ in range(n_pairs)]

    for sid in tqdm(train_ids, desc="Computing amp stats", leave=False):
        sig = normalized_data_map[sid]
        for pair_j in range(n_pairs):
            window = sig[:lookback, pair_j]
            if len(window) < lookback:
                continue
            fft_c = scipy.fft.rfft(window, n=lookback)
            amps_at_bins = np.abs(fft_c[fixed_fft_bin_indices])
            for fi in range(n_freqs):
                amps_list[pair_j][fi].append(float(amps_at_bins[fi]))

    amp_means = np.zeros((n_pairs, n_freqs), dtype=np.float32)
    amp_stds  = np.ones( (n_pairs, n_freqs), dtype=np.float32)
    for pj in range(n_pairs):
        for fi in range(n_freqs):
            vals = amps_list[pj][fi]
            if vals:
                amp_means[pj, fi] = float(np.mean(vals))
                s = float(np.std(vals)) if len(vals) > 1 else 1.0
                amp_stds[pj, fi]  = max(s, 1e-6)
    return amp_means, amp_stds


# ── Step 3: Average baseline energy profile ───────────────────────────────────
def compute_baseline_energy_profile(
    normalized_data_map: dict,
    train_ids: list,
    amp_means: np.ndarray,
    amp_stds: np.ndarray,
    fixed_fft_bin_indices: np.ndarray = FIXED_FFT_BIN_INDICES,
    lookback: int = LOOKBACK_POINTS,
) -> torch.Tensor:
    """Returns tensor [66, 256]."""
    baseline_ids = [s for s in train_ids if "baseline" in s]
    amp_m = torch.from_numpy(amp_means)
    amp_s = torch.from_numpy(amp_stds)
    profiles = []
    for sid in tqdm(baseline_ids, desc="Baseline energy profile", leave=False):
        sig  = torch.from_numpy(normalized_data_map[sid]).float()
        fftc = torch.fft.rfft(sig[:lookback], n=lookback, dim=0)
        amps = torch.abs(fftc[fixed_fft_bin_indices]).T       # [66, 256]
        norm_amps = (amps - amp_m) / amp_s
        profiles.append(torch.abs(norm_amps))
    if profiles:
        return torch.stack(profiles, dim=0).mean(dim=0)       # [66, 256]
    return torch.zeros(NUM_DATA_COLUMNS, len(fixed_fft_bin_indices))


# ── Step 4: Global max delta-E ─────────────────────────────────────────────────
def compute_global_max_delta_e(
    normalized_data_map: dict,
    train_ids: list,
    amp_means: np.ndarray,
    amp_stds: np.ndarray,
    average_baseline_energy_profile: torch.Tensor,
    propagation_pair_indices: torch.Tensor,
    fixed_fft_bin_indices: np.ndarray = FIXED_FFT_BIN_INDICES,
    lookback: int = LOOKBACK_POINTS,
) -> float:
    amp_m = torch.from_numpy(amp_means)
    amp_s = torch.from_numpy(amp_stds)
    max_val = 0.0
    for sid in tqdm(train_ids, desc="Computing global max ΔE", leave=False):
        sig  = torch.from_numpy(normalized_data_map[sid]).float()
        fftc = torch.fft.rfft(sig[:lookback], n=lookback, dim=0)
        amps = torch.abs(fftc[fixed_fft_bin_indices]).T       # [66, 256]
        norm_amps = (amps - amp_m) / amp_s
        delta_e = (torch.abs(norm_amps) - average_baseline_energy_profile
                   ).mean(dim=-1).clamp(min=0)                # [66]
        max_val = max(max_val, delta_e[propagation_pair_indices].max().item())
    return max(max_val, 1e-6)


# ── Edge/graph topology helpers ───────────────────────────────────────────────


def get_propagation_edge_index_and_col_idxs(
    num_transducers: int = NUM_TRANSDUCERS,
) -> tuple[torch.Tensor, list, torch.Tensor]:
    """
    Top↔bottom propagation paths only (36 pairs, 72 directed edges).

    BUG FIX: iterate over sorted() lists, not sets, to guarantee deterministic
    ordering. The DirectPathAttenuationGNN reshape view(B, 36, 2, 1) depends on
    edges being in a consistent pair order across runs.
    """
    top    = sorted(range(num_transducers // 2))           # [0,1,2,3,4,5]
    bottom = sorted(range(num_transducers // 2, num_transducers))  # [6,7,8,9,10,11]

    all_pairs   = list(itertools.combinations(range(num_transducers), 2))
    pair_to_col = {p: i for i, p in enumerate(all_pairs)}

    edges, col_idxs = [], []
    for i in top:
        for j in bottom:
            for u, v in [(i, j), (j, i)]:   # both directions; adjacent → view works
                edges.append([u, v])
                col_idxs.append(pair_to_col[tuple(sorted((u, v)))])

    prop_ei      = torch.tensor(edges, dtype=torch.long).t().contiguous()
    unique_cols  = torch.tensor(sorted(set(col_idxs)), dtype=torch.long)
    return prop_ei, col_idxs, unique_cols


def get_inv_edge_feature_col_idxs(
    static_edge_index: torch.Tensor,
    num_transducers: int = NUM_TRANSDUCERS,
) -> np.ndarray:
    """Map every directed edge (u,v) → data column index of pair {u,v}."""
    all_pairs   = list(itertools.combinations(range(num_transducers), 2))
    pair_to_col = {p: i for i, p in enumerate(all_pairs)}
    return np.array(
        [pair_to_col[tuple(sorted((
            static_edge_index[0, i].item(),
            static_edge_index[1, i].item()
        )))] for i in range(static_edge_index.shape[1])],
        dtype=np.int64,
    )


# ── Master build function ──────────────────────────────────────────────────────
def build_all_stats(raw_data_map: dict, train_ids: list) -> dict:
    """
    Build all statistics for a given split from the raw data map.
    Call once per split at the top of each main_*.py script.
    Returns stats dict; always use stats["normalized_data_map"] for datasets.
    """
    from utils.data_loader import get_k_graph_edge_index

    print("─ Step 0+1: Baseline subtraction + time-domain z-normalization …")
    norm_data, _, _, _ = build_normalized_differential_data(raw_data_map, train_ids)

    print("─ Step 2: Per-(pair,freq) amplitude statistics …")
    amp_means, amp_stds = compute_amp_stats(norm_data, train_ids)

    print("─ Step 3: Average baseline energy profile …")
    avg_bl = compute_baseline_energy_profile(norm_data, train_ids, amp_means, amp_stds)

    print("─ Building propagation edge index (sorted, deterministic) …")
    prop_ei, prop_col_idxs, prop_unique = get_propagation_edge_index_and_col_idxs()

    print("─ Step 4: Global max ΔE …")
    g_max = compute_global_max_delta_e(
        norm_data, train_ids, amp_means, amp_stds, avg_bl, prop_unique
    )
    print(f"  global_max_delta_e = {g_max:.6f}")

    k12          = get_k_graph_edge_index(NUM_TRANSDUCERS, self_loops=False)
    inv_col_idxs = get_inv_edge_feature_col_idxs(k12)

    return dict(
        normalized_data_map             = norm_data,   # ← use this in ALL datasets
        fixed_fft_bin_indices           = FIXED_FFT_BIN_INDICES,
        amp_means                       = amp_means,   # [66, 256]
        amp_stds                        = amp_stds,    # [66, 256]
        average_baseline_energy_profile = avg_bl,      # Tensor [66, 256]
        global_max_delta_e              = g_max,
        propagation_edge_index          = prop_ei,     # [2, 72]
        propagation_col_idxs            = prop_col_idxs,
        propagation_pair_indices        = prop_unique, # Tensor [36]
        k12_edge_index                  = k12,         # [2, 132]
        inv_edge_feature_col_idxs       = inv_col_idxs,
        lookback_fft                    = LOOKBACK_POINTS,
        num_fft_bins                    = N_ATTENTION_FREQS,
    )