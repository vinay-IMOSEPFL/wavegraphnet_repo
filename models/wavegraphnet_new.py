import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GATv2Conv, MessagePassing, global_mean_pool
from torch_geometric.data import Data


class DynamicWeightedLoss(nn.Module):
    """
    Implements learnable loss weighting.
    Prevents one loss from dominating the gradient flow.
    """

    def __init__(self, num_losses=2):
        super().__init__()
        self.params = nn.Parameter(torch.ones(num_losses))

    def forward(self, losses):
        weighted_losses = []
        for i, loss in enumerate(losses):
            weighted_losses.append(
                loss / (2 * self.params[i] ** 2) + torch.log(self.params[i] ** 2)
            )
        return sum(weighted_losses)


class InverseGNN(nn.Module):
    def __init__(self, node_in, edge_in, hidden, out_dim=2):
        super().__init__()
        self.node_enc = nn.Linear(node_in, hidden)
        self.edge_enc = nn.Linear(edge_in, hidden)

        # Using GATv2 for more robust attention over sensor paths
        self.conv1 = GATv2Conv(hidden, hidden, edge_dim=hidden, heads=4, concat=True)
        self.conv2 = GATv2Conv(
            hidden * 4, hidden, edge_dim=hidden, heads=1, concat=False
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
            nn.Sigmoid(),  # Constrains coordinates to [0, 1]
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.node_enc(x)
        edge_attr = self.edge_enc(edge_attr)

        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = self.conv2(x, edge_index, edge_attr)

        # Graph-level pooling to get global damage feature
        out = global_mean_pool(x, data.batch)
        return self.decoder(out)


class ForwardPhysicsGNN(nn.Module):
    """
    Predicts path-wise energy deviation (Delta E).

    FIX: Added num_propagation_pairs so the output can be reshaped from
    [batch * num_directed_edges] -> [batch, num_pairs] by averaging the
    two directed versions (A->B and B->A) of each undirected path.
    This matches the shape of delta_e_true which has one value per unique
    undirected pair (66 for a 12-node fully-connected graph).
    """

    def __init__(self, edge_in=6, hidden=128, num_propagation_pairs=66):
        super().__init__()
        # FIX: store so forward() can reshape correctly
        self.num_propagation_pairs = num_propagation_pairs

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )
        # Residual Message Passing layers
        self.res_linear = nn.Linear(hidden, hidden)
        self.out_layer = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # One Delta E value per directed edge
        )

    def forward(self, data, pred_coords):
        row, col = data.edge_index
        s_coords = data.x[row]
        r_coords = data.x[col]

        # Dynamic geometry calculation based on inverse prediction
        d_coords = pred_coords[data.batch[row]]

        # Geometric features
        vec = s_coords - r_coords
        dist_sr = torch.norm(vec, dim=-1, keepdim=True)
        dist_sd = torch.norm(s_coords - d_coords, dim=-1, keepdim=True)
        dist_rd = torch.norm(r_coords - d_coords, dim=-1, keepdim=True)

        # Path-to-damage perpendicular distance
        l2 = torch.sum(vec**2, dim=-1, keepdim=True) + 1e-8
        t = torch.sum((d_coords - s_coords) * vec, dim=-1, keepdim=True) / l2
        proj = s_coords + t.clamp(0, 1) * vec
        dist_path = torch.norm(d_coords - proj, dim=-1, keepdim=True)

        f_feats = torch.cat([vec, dist_sr, dist_sd, dist_rd, dist_path], dim=-1)

        h_e = self.edge_mlp(f_feats)
        h_e = F.relu(self.res_linear(h_e) + h_e)

        # Shape: [batch * num_directed_edges]  e.g. [32 * 132] = [4224]
        edge_preds = self.out_layer(h_e).squeeze(-1)

        # FIX: The graph has num_propagation_pairs*2 directed edges (132 = 66*2).
        # delta_e_true has shape [batch, num_propagation_pairs] (one per unique path).
        # Reshape to [batch, num_pairs, 2] and average the two directions so output
        # matches target shape [batch, num_propagation_pairs].
        batch_size = pred_coords.shape[0]
        edge_preds = edge_preds.view(batch_size, self.num_propagation_pairs, 2)
        return edge_preds.mean(dim=2)  # [batch, num_propagation_pairs]