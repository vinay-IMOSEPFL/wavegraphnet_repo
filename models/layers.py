# models/layers.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    MessagePassing, global_mean_pool, global_add_pool,
)


def global_max_pool_safe(x: torch.Tensor, batch: torch.Tensor, size: int = None) -> torch.Tensor:
    """
    Autograd-safe global max pooling.
    Uses a per-graph masked max — no dependency on torch_scatter CUDA kernels.
    """
    if size is None:
        size = int(batch.max().item()) + 1
    pieces = []
    for g in range(size):
        mask = (batch == g)
        pieces.append(x[mask].max(dim=0).values if mask.any() else x.new_zeros(x.size(-1)))
    return torch.stack(pieces, dim=0)


class NodeEncoder(nn.Module):
    def __init__(self, raw_node_feat_dim, embedding_dim, hidden_dim=None, num_layers=1):
        super().__init__()
        if num_layers == 0:
            self.encoder = nn.Identity()
        elif num_layers == 1:
            self.encoder = nn.Linear(raw_node_feat_dim, embedding_dim)
        else:
            hd = hidden_dim or (raw_node_feat_dim + embedding_dim) // 2
            layers = [nn.Linear(raw_node_feat_dim, hd), nn.ReLU()]
            for _ in range(num_layers - 2):
                layers += [nn.Linear(hd, hd), nn.ReLU()]
            layers.append(nn.Linear(hd, embedding_dim))
            self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class GraphDecoder(nn.Module):
    def __init__(self, final_node_embedding_dim, mlp_hidden_dim, output_dim=2,
                 pooling_type='max', num_decoder_mlp_layers=3, dropout_rate=0.2):
        super().__init__()
        if pooling_type == 'mean':
            self.pooling = global_mean_pool
        elif pooling_type == 'add':
            self.pooling = global_add_pool
        elif pooling_type == 'max':
            self.pooling = global_max_pool_safe   # notebook uses max; safe on CPU+CUDA
        else:
            raise ValueError(f"Unsupported pooling: {pooling_type}")

        layers = [nn.Linear(final_node_embedding_dim, mlp_hidden_dim),
                  nn.ReLU(), nn.Dropout(dropout_rate)]
        for _ in range(num_decoder_mlp_layers - 2):
            layers += [nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
                       nn.ReLU(), nn.Dropout(dropout_rate)]
        layers.append(nn.Linear(mlp_hidden_dim, output_dim))
        self.readout_mlp = nn.Sequential(*layers)

    def forward(self, final_node_embeddings, batch_vector):
        graph_emb = self.pooling(final_node_embeddings, batch_vector)
        return self.readout_mlp(graph_emb)


class RichEdgeConv(MessagePassing):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_channels, out_channels, dropout_rate=0.2):
        super().__init__(aggr='mean')
        self.mlp_message = nn.Sequential(
            nn.Linear(node_feat_dim * 2 + edge_feat_dim, hidden_channels), nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_channels, hidden_channels), nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.mlp_update = nn.Sequential(
            nn.Linear(node_feat_dim + hidden_channels, out_channels), nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        return self.mlp_message(torch.cat([x_j, x_i, edge_attr], dim=1))

    def update(self, aggr_out, x):
        return self.mlp_update(torch.cat([x, aggr_out], dim=1))