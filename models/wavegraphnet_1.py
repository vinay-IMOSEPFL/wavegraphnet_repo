import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data as PyGData
from torch_geometric.nn import GATConv, MessagePassing
from models.layers import NodeEncoder, GraphDecoder


class EdgeEncoderWithAttention(nn.Module):
    def __init__(
        self,
        num_freqs,
        feature_dim_per_freq,
        static_feat_dim,
        final_embedding_dim,
        dropout_rate=0.2,
    ):
        super().__init__()
        self.static_feat_dim = static_feat_dim
        self.feature_dim_per_freq = feature_dim_per_freq
        self.feature_processor = nn.Sequential(
            nn.Linear(feature_dim_per_freq, 64), nn.ReLU(), nn.Dropout(dropout_rate)
        )
        self.attention_mlp = nn.Sequential(
            nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 1)
        )
        self.combiner_mlp = nn.Sequential(
            nn.Linear(64 + static_feat_dim, final_embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(final_embedding_dim, final_embedding_dim),
        )

    def forward(self, x_edges):
        static_feats = x_edges[:, : self.static_feat_dim]
        freq_features_flat = x_edges[:, self.static_feat_dim :]
        freq_features = freq_features_flat.view(
            freq_features_flat.shape[0], -1, self.feature_dim_per_freq
        )
        processed_freq_feats = self.feature_processor(freq_features)
        attention_scores = self.attention_mlp(processed_freq_feats)
        attention_weights = F.softmax(attention_scores, dim=1)
        dynamic_embedding = (attention_weights * processed_freq_feats).sum(dim=1)
        combined_final = torch.cat([dynamic_embedding, static_feats], dim=1)
        final_embedding = self.combiner_mlp(combined_final)
        return final_embedding


class GNNProcessor_GAT(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_gnn_layers: int,
        num_attention_heads: int,
        dropout_rate: float,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout_rate = dropout_rate

        self.convs.append(
            GATConv(hidden_dim, hidden_dim, heads=num_attention_heads, concat=True)
        )
        self.norms.append(nn.LayerNorm(hidden_dim * num_attention_heads))

        for _ in range(num_gnn_layers - 2):
            self.convs.append(
                GATConv(
                    hidden_dim * num_attention_heads,
                    hidden_dim,
                    heads=num_attention_heads,
                    concat=True,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim * num_attention_heads))

        self.convs.append(
            GATConv(hidden_dim * num_attention_heads, hidden_dim, heads=1, concat=False)
        )
        self.norms.append(nn.LayerNorm(hidden_dim))

    def forward(self, node_embeds, edge_index, edge_embeds):
        row, col = edge_index
        edge_messages = torch_geometric.utils.scatter(
            edge_embeds, col, dim=0, reduce="mean"
        )
        x = node_embeds + edge_messages

        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index)
            x = self.norms[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        return x


class GNN_inv_HierarchicalAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        raw_node_feat_dim: int,
        num_attention_freqs: int,
        num_gnn_proc_layers: int,
        gat_attention_heads: int,
        decoder_mlp_hidden_dim: int,
        final_output_dim: int,
        decoder_pooling_type: str = "max",
        num_decoder_mlp_layers: int = 3,
        decoder_dropout_rate: float = 0.2,
    ):

        super().__init__()
        self.node_encoder = NodeEncoder(
            raw_node_feat_dim=raw_node_feat_dim, embedding_dim=hidden_dim
        )
        self.edge_encoder = EdgeEncoderWithAttention(
            num_freqs=num_attention_freqs,
            feature_dim_per_freq=2,
            static_feat_dim=3,
            final_embedding_dim=hidden_dim,
            dropout_rate=decoder_dropout_rate,
        )
        self.gnn_processor = GNNProcessor_GAT(
            hidden_dim=hidden_dim,
            num_gnn_layers=num_gnn_proc_layers,
            num_attention_heads=gat_attention_heads,
            dropout_rate=decoder_dropout_rate,
        )
        self.graph_decoder = GraphDecoder(
            final_node_embedding_dim=hidden_dim,
            mlp_hidden_dim=decoder_mlp_hidden_dim,
            output_dim=final_output_dim,
            pooling_type=decoder_pooling_type,
            num_decoder_mlp_layers=num_decoder_mlp_layers,
            dropout_rate=decoder_dropout_rate,
        )

    def forward(self, data: PyGData):
        x_nodes_raw, edge_index, x_edges_raw, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )
        node_embeddings_initial = self.node_encoder(x_nodes_raw)
        edge_embeddings_initial = self.edge_encoder(x_edges_raw)
        node_embeddings_final = self.gnn_processor(
            node_embeds=node_embeddings_initial,
            edge_index=edge_index,
            edge_embeds=edge_embeddings_initial,
        )

        # No sigmoid: model must be able to predict -0.001 (undamaged sentinel)
        # which is outside [0,1]. Sigmoid would make that impossible and cause
        # the model to collapse to predicting 0.5 for all samples.
        return self.graph_decoder(node_embeddings_final, batch)


class SimpleInteractionLayer(MessagePassing):
    def __init__(self, hidden_dim: int):
        super().__init__(aggr="mean")
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x_n, x_e, edge_index):
        sender_nodes = x_n[edge_index[0]]
        receiver_nodes = x_n[edge_index[1]]
        edge_mlp_input = torch.cat([sender_nodes, receiver_nodes, x_e], dim=1)
        edge_update = self.edge_mlp(edge_mlp_input)
        x_e_updated = x_e + edge_update
        node_update = self.propagate(edge_index, x=x_n, edge_attr=x_e_updated)
        x_n_updated = x_n + node_update
        return x_n_updated, x_e_updated

    def update(self, aggr_out, x):
        node_mlp_input = torch.cat([x, aggr_out], dim=1)
        return self.node_mlp(node_mlp_input)


class DirectPathAttenuationGNN(nn.Module):
    def __init__(
        self,
        raw_node_feat_dim: int = 2,
        physical_edge_feat_dim: int = 6,
        hidden_dim: int = 128,
        num_propagation_pairs: int = 36,
        num_interaction_layers: int = 4,
    ):
        super().__init__()
        self.num_propagation_pairs = num_propagation_pairs
        self.node_encoder = nn.Linear(raw_node_feat_dim, hidden_dim)
        self.edge_encoder = nn.Sequential(
            nn.Linear(physical_edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.processor = nn.ModuleList(
            [
                SimpleInteractionLayer(hidden_dim=hidden_dim)
                for _ in range(num_interaction_layers)
            ]
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),   # geometric influence target ∈ (0,1]
        )

    def forward(self, graph_sensors_fwd: PyGData, damage_locs: torch.Tensor):
        batch_size = damage_locs.shape[0]
        row, col = graph_sensors_fwd.edge_index
        source_nodes_coords = graph_sensors_fwd.x[row]
        dest_nodes_coords = graph_sensors_fwd.x[col]
        damage_locs_expanded = damage_locs[graph_sensors_fwd.batch[row]]

        vec = source_nodes_coords - dest_nodes_coords

        # CRITICAL FIX 2: Epsilon-smoothed geometric distances to prevent NaN gradients
        eps = 1e-8

        edge_length = torch.sqrt(vec.pow(2).sum(dim=-1, keepdim=True) + eps)

        p1, p2, p3 = source_nodes_coords, dest_nodes_coords, damage_locs_expanded
        l2 = (p2 - p1).pow(2).sum(dim=-1, keepdim=True).clamp(min=eps)
        t = torch.sum((p3 - p1) * (p2 - p1), dim=-1, keepdim=True) / l2
        t = t.clamp(0, 1)
        projection = p1 + t * (p2 - p1)

        dist_from_damage = torch.sqrt(
            (p3 - projection).pow(2).sum(dim=-1, keepdim=True) + eps
        )
        dist_transmitter_to_damage = torch.sqrt(
            (source_nodes_coords - damage_locs_expanded)
            .pow(2)
            .sum(dim=-1, keepdim=True)
            + eps
        )
        dist_receiver_to_damage = torch.sqrt(
            (dest_nodes_coords - damage_locs_expanded).pow(2).sum(dim=-1, keepdim=True)
            + eps
        )

        physical_edge_features = torch.cat(
            [
                vec,
                edge_length,
                dist_from_damage,
                dist_transmitter_to_damage,
                dist_receiver_to_damage,
            ],
            dim=1,
        )
        h_n = self.node_encoder(graph_sensors_fwd.x)
        h_e = self.edge_encoder(physical_edge_features)

        for layer in self.processor:
            h_n, h_e = layer(h_n, h_e, graph_sensors_fwd.edge_index)

        predicted_delta_e_directed = self.decoder(h_e)
        delta_e_reshaped = predicted_delta_e_directed.view(
            batch_size, self.num_propagation_pairs, 2, 1
        )
        final_delta_e_map = delta_e_reshaped.mean(dim=2).squeeze(-1)

        return final_delta_e_map