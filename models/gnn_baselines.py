import torch
import torch.nn as nn
from torch_geometric.data import Data as PyGData
from models.layers import NodeEncoder, GraphDecoder, RichEdgeConv
from models.wavegraphnet import EdgeEncoderWithAttention


class SimpleEdgeEncoder(nn.Module):
    def __init__(
        self,
        raw_edge_feat_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int = 3,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        layers = [
            nn.Linear(raw_edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_rate),
        ]
        for _ in range(num_layers - 2):
            layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(dropout_rate),
                ]
            )
        layers.append(nn.Linear(hidden_dim, embedding_dim))
        self.encoder_mlp = nn.Sequential(*layers)

    def forward(self, x_edges: torch.Tensor) -> torch.Tensor:
        return self.encoder_mlp(x_edges)


class GNNProcessor_MLP(nn.Module):
    def __init__(self, hidden_dim, num_gnn_layers, dropout_rate=0.2):
        super().__init__()
        self.convs = nn.ModuleList()
        for _ in range(num_gnn_layers):
            self.convs.append(
                RichEdgeConv(
                    node_feat_dim=hidden_dim,
                    edge_feat_dim=hidden_dim,
                    hidden_channels=hidden_dim,
                    out_channels=hidden_dim,
                    dropout_rate=dropout_rate,
                )
            )

    def forward(self, node_embeds, edge_index, edge_embeds):
        x_nodes = node_embeds
        for conv_layer in self.convs:
            x_nodes = conv_layer(x_nodes, edge_index, edge_embeds)
        return x_nodes


class FlexibleGNN(nn.Module):
    def __init__(
        self,
        encoder_type: str,
        processor_type: str,
        raw_node_feat_dim,
        raw_edge_feat_dim,
        num_attention_freqs,
        hidden_dim,
        num_gnn_proc_layers,
        gat_attention_heads,
        decoder_mlp_hidden_dim,
        final_output_dim,
        decoder_pooling_type,
        num_decoder_mlp_layers,
        decoder_dropout_rate,
    ):
        super().__init__()
        self.node_encoder = NodeEncoder(raw_node_feat_dim, hidden_dim)

        if encoder_type == "attention":
            self.edge_encoder = EdgeEncoderWithAttention(
                num_attention_freqs, 2, 3, hidden_dim, decoder_dropout_rate
            )
        elif encoder_type == "simple_mlp":
            self.edge_encoder = SimpleEdgeEncoder(
                raw_edge_feat_dim, hidden_dim, hidden_dim * 2, 4, decoder_dropout_rate
            )
        else:
            raise ValueError(f"Unknown encoder_type: '{encoder_type}'.")

        if processor_type == "mlp":
            self.gnn_processor = GNNProcessor_MLP(
                hidden_dim, num_gnn_proc_layers, decoder_dropout_rate
            )
        else:
            # Assumes GNNProcessor_GAT is imported or available if added here
            raise ValueError(f"Unknown processor_type: '{processor_type}'.")

        self.graph_decoder = GraphDecoder(
            hidden_dim,
            decoder_mlp_hidden_dim,
            final_output_dim,
            decoder_pooling_type,
            num_decoder_mlp_layers,
            decoder_dropout_rate,
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
            node_embeddings_initial, edge_index, edge_embeddings_initial
        )
        predictions = self.graph_decoder(node_embeddings_final, batch)
        return predictions
