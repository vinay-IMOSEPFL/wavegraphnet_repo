# utils/data_loader.py
import torch
import numpy as np
import scipy.fft
from torch_geometric.data import Data as PyGData
from torch.utils.data import Dataset as TorchDataset
import itertools

TRANSDUCER_COORDS = {
    1: [0.9, 0.94], 2: [0.74, 0.94], 3: [0.58, 0.94], 4: [0.42, 0.94],
    5: [0.26, 0.94], 6: [0.1, 0.94], 7: [0.9, 0.06], 8: [0.74, 0.06],
    9: [0.58, 0.06], 10: [0.42, 0.06], 11: [0.26, 0.06], 12: [0.1, 0.06],
}

DAMAGE_LABELS = {
    "D1": [0.1, 0.83], "D2": [0.13, 0.83], "D3": [0.1, 0.8],
    "D4": [0.13, 0.8], "D5": [0.5, 0.854], "D6": [0.53, 0.854],
    "D7": [0.5, 0.824], "D8": [0.53, 0.824], "D9": [0.36, 0.69],
    "D10": [0.39, 0.69], "D11": [0.36, 0.66], "D12": [0.39, 0.66],
    "D13": [0.64, 0.55], "D14": [0.67, 0.55], "D15": [0.64, 0.52],
    "D16": [0.67, 0.52], "D17": [0.26, 0.39], "D18": [0.29, 0.39],
    "D19": [0.26, 0.36], "D20": [0.29, 0.36], "D21": [0.87, 0.41],
    "D22": [0.9, 0.41], "D23": [0.87, 0.38], "D24": [0.9, 0.38],
    "D25": [0.5, 0.18], "D26": [0.53, 0.18], "D27": [0.5, 0.15],
    "D28": [0.53, 0.15],
}


def get_k_graph_edge_index(num_nodes, self_loops=False):
    edges = list(itertools.combinations(range(num_nodes), 2))
    edge_list = []
    for u, v in edges:
        edge_list.append([u, v])
        edge_list.append([v, u])
    if self_loops:
        for i in range(num_nodes):
            edge_list.append([i, i])
    if not edge_list:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edge_list, dtype=torch.long).t().contiguous()


def parse_damage_label(damage_string: str) -> str:
    if "baseline" in damage_string:
        return "undamaged"
    return damage_string.split("_")[0]


class CoupledModelDataset(TorchDataset):
    """
    Dataset for WaveGraphNet (inverse + forward branches).

    amp_means / amp_stds must be shape [num_pairs, num_freqs] = [66, 256].
    average_baseline_energy_profile must be shape [66, 256].
    inv_edge_feature_col_idxs must be length == num_directed_edges (132).
    fwd_propagation_col_idxs holds column indices for the 36 propagation pairs.
    """

    def __init__(
        self, data_map, sample_id_list,
        inv_static_edge_index, inv_edge_feature_col_idxs,
        fwd_propagation_col_idxs, fixed_fft_bin_indices,
        amp_means, amp_stds, lookback_fft,
        average_baseline_energy_profile, global_max_delta_e,
        fwd_prop_edge_index=None,   # [2, 72] propagation edges for geometric target
        geometric_sigma=0.1,        # scale in normalized coords (0.1 = 50mm)
    ):
        self.data_map = data_map
        self.sample_id_list = sample_id_list
        self.node_coords = torch.tensor(
            np.array([TRANSDUCER_COORDS[i + 1] for i in range(12)]), dtype=torch.float
        )
        self.inv_edge_index = inv_static_edge_index
        self.inv_col_idxs = torch.as_tensor(inv_edge_feature_col_idxs, dtype=torch.long)
        self.lookback_fft = lookback_fft
        self.fft_bins = torch.as_tensor(fixed_fft_bin_indices, dtype=torch.long)
        self.num_freqs = len(fixed_fft_bin_indices)

        # Per-(pair, freq) stats  [66, 256]
        self.amp_means = torch.tensor(amp_means, dtype=torch.float32)
        self.amp_stds  = torch.tensor(amp_stds,  dtype=torch.float32)

        self.global_max_delta_e = max(float(global_max_delta_e), 1e-6)
        self.prop_pair_indices = torch.tensor(
            sorted(set(fwd_propagation_col_idxs)), dtype=torch.long
        )
        self.avg_baseline = average_baseline_energy_profile  # [66, 256]

        # Geometric influence target (Option D)
        # If fwd_prop_edge_index provided, use geometric target instead of delta_e
        self.geometric_sigma = geometric_sigma
        if fwd_prop_edge_index is not None:
            # prop_ei has 72 directed edges (36 pairs × 2); even indices = one direction
            self.prop_src = fwd_prop_edge_index[0, 0::2].tolist()  # 36 src transducer idxs
            self.prop_dst = fwd_prop_edge_index[1, 0::2].tolist()  # 36 dst transducer idxs
            self.use_geometric_target = True
        else:
            self.prop_src = None
            self.prop_dst = None
            self.use_geometric_target = False

    def __len__(self):
        return len(self.sample_id_list)

    def __getitem__(self, idx):
        sample_id = self.sample_id_list[idx]
        sig = torch.from_numpy(self.data_map[sample_id]).float()
        num_pairs = sig.shape[1]

        fft_full = scipy.fft.rfft(
            sig[:self.lookback_fft, :].numpy(), n=self.lookback_fft, axis=0
        )
        bins = self.fft_bins.numpy()
        amps   = torch.from_numpy(np.abs(fft_full[bins, :]).astype(np.float32)).T    # [66, 256]
        phases = torch.from_numpy(np.angle(fft_full[bins, :]).astype(np.float32)).T  # [66, 256]

        # Per-(pair, freq) normalisation
        norm_amps = (amps - self.amp_means) / self.amp_stds   # [66, 256]
        full_freq_profile = torch.stack([norm_amps, phases], dim=-1)  # [66, 256, 2]
        flat = full_freq_profile.view(num_pairs, -1)                   # [66, 512]

        # Build inverse graph edge attributes
        row, col = self.inv_edge_index
        dist = (self.node_coords[row] - self.node_coords[col]).norm(dim=-1, keepdim=True)
        vec  = self.node_coords[row] - self.node_coords[col]
        sp   = torch.cat([dist, vec], dim=1)      # [132, 3]
        freq = flat[self.inv_col_idxs]             # [132, 512]
        edge_attr_inv = torch.cat([sp, freq], dim=1)  # [132, 515]

        data_inv = PyGData(
            x=self.node_coords, edge_index=self.inv_edge_index, edge_attr=edge_attr_inv,
        )

        dmg = parse_damage_label(sample_id)
        xd, yd = -0.001, -0.001
        if dmg != "undamaged" and dmg in DAMAGE_LABELS:
            xd, yd = DAMAGE_LABELS[dmg]
        y_true = torch.tensor([[xd, yd]], dtype=torch.float)

        # Forward branch target
        if self.use_geometric_target:
            # Option D: geometric path influence — purely from damage location + geometry
            # influence_ij = 1 / (1 + excess_ij / sigma)
            # excess_ij = ||p-ri|| + ||p-rj|| - ||rj-ri||  (extra travel distance)
            # Range [0,1]: 1 when damage is on path, decays as damage moves away.
            # Works for undamaged sentinel too (gives low influence, masked in training).
            coords = self.node_coords  # [12, 2]
            p = torch.tensor([xd, yd], dtype=torch.float32)  # damage coord
            influences = []
            for s, d in zip(self.prop_src, self.prop_dst):
                ri = coords[s]; rj = coords[d]
                excess = ((p-ri).norm() + (p-rj).norm() - (rj-ri).norm()).clamp(min=0)
                influences.append(1.0 / (1.0 + excess / self.geometric_sigma))
            delta_e_norm = torch.stack(influences)                        # [36] in (0,1]
        else:
            # Option A: signed signal-derived delta-E (remove clamp)
            cur_energy   = torch.abs(norm_amps)                           # [66, 256]
            delta_e      = (cur_energy - self.avg_baseline).mean(dim=-1)  # [66] signed
            delta_e_prop = delta_e[self.prop_pair_indices]                # [36]
            delta_e_norm = delta_e_prop / self.global_max_delta_e

        return {
            "data_inv": data_inv,
            "delta_e_true": delta_e_norm,
            "y_true": y_true,
            "sample_id": sample_id,
        }


class StandardGraphDataset(TorchDataset):
    """Dataset for baseline GNN models."""

    def __init__(
        self, data_map, sample_id_list, static_edge_index, edge_feature_col_idxs,
        fixed_fft_bin_indices, amp_means, amp_stds, lookback_fft,
    ):
        self.data_map    = data_map
        self.sample_ids  = sample_id_list
        self.node_coords = torch.tensor(
            np.array([TRANSDUCER_COORDS[i + 1] for i in range(12)]), dtype=torch.float
        )
        self.edge_index  = static_edge_index
        self.col_idxs    = torch.as_tensor(edge_feature_col_idxs, dtype=torch.long)
        self.lookback    = lookback_fft
        self.fft_bins    = torch.as_tensor(fixed_fft_bin_indices, dtype=torch.long)
        self.num_freqs   = len(fixed_fft_bin_indices)
        self.amp_means   = torch.tensor(amp_means, dtype=torch.float32)  # [66, 256]
        self.amp_stds    = torch.tensor(amp_stds,  dtype=torch.float32)  # [66, 256]

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        sig       = torch.from_numpy(self.data_map[sample_id]).float()
        num_pairs = sig.shape[1]

        fft_full = scipy.fft.rfft(
            sig[:self.lookback, :].numpy(), n=self.lookback, axis=0
        )
        bins = self.fft_bins.numpy()
        amps   = torch.from_numpy(np.abs(fft_full[bins, :]).astype(np.float32)).T  # [66, 256]
        phases = torch.from_numpy(np.angle(fft_full[bins, :]).astype(np.float32)).T

        norm_amps = (amps - self.amp_means) / self.amp_stds
        flat = torch.stack([norm_amps, phases], dim=-1).view(num_pairs, -1)  # [66, 512]

        row, col = self.edge_index
        dist = (self.node_coords[row] - self.node_coords[col]).norm(dim=-1, keepdim=True)
        vec  = self.node_coords[row] - self.node_coords[col]
        sp   = torch.cat([dist, vec], dim=1)
        freq = flat[self.col_idxs]
        edge_attr = torch.cat([sp, freq], dim=1)

        data = PyGData(x=self.node_coords, edge_index=self.edge_index, edge_attr=edge_attr)

        dmg = parse_damage_label(sample_id)
        xd, yd = -0.001, -0.001
        if dmg != "undamaged" and dmg in DAMAGE_LABELS:
            xd, yd = DAMAGE_LABELS[dmg]
        data.y = torch.tensor([[xd, yd]], dtype=torch.float)

        return data