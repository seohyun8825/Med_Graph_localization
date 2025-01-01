import torch
import torch.nn as nn
from model.layers_t7 import Embedding, VisualProjection, FeatureEncoder, ConditionedPredictor
from transformers import AdamW, get_linear_schedule_with_warmup


def build_optimizer_and_scheduler(model, configs):
    no_decay = ['bias', 'layer_norm', 'LayerNorm']  # no decay for parameters of layer norm and bias
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=configs.init_lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, configs.num_train_steps * configs.warmup_proportion,
                                                configs.num_train_steps)
    return optimizer, scheduler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, TransformerConv
from torch_geometric.data import Data

class VSLNet(nn.Module):
    def __init__(self, configs, word_vectors):
        super(VSLNet, self).__init__()
        self.configs = configs
        # Embedding Layers
        self.embedding_net = Embedding(
            num_words=configs.word_size,
            num_chars=configs.char_size,
            out_dim=configs.dim,
            word_dim=configs.word_dim,
            char_dim=configs.char_dim,
            word_vectors=word_vectors,
            drop_rate=configs.drop_rate
        )
        self.video_proj = VisualProjection(configs.video_feature_dim, configs.dim)
        self.feature_encoder = FeatureEncoder(dim=configs.dim, num_heads=configs.num_heads, kernel_size=7, num_layers=4,
                                              max_pos_len=configs.max_pos_len, drop_rate=configs.drop_rate)

        # Graph Transformer Layers
        '''
        self.graph_layers = nn.ModuleList([
            TransformerConv(configs.dim, configs.dim // configs.num_heads, heads=configs.num_heads)
            for _ in range(4)
        ])
        '''
        self.temporal_graph = nn.ModuleList([
            TransformerConv(configs.dim, configs.dim // configs.num_heads, heads=configs.num_heads)
            for _ in range(4)
        ])
        self.semantic_graph = nn.ModuleList([
            TransformerConv(configs.dim, configs.dim // configs.num_heads, heads=configs.num_heads)
            for _ in range(4)
        ])
        self.query_graph = nn.ModuleList([
            TransformerConv(configs.dim, configs.dim // configs.num_heads, heads=configs.num_heads)
            for _ in range(4)
        ])
        self.attention_layer = nn.MultiheadAttention(embed_dim=configs.dim, num_heads=configs.num_heads)
        # Start-End Time Prediction
        self.start_predictor = nn.Linear(configs.dim, 1)
        self.end_predictor = nn.Linear(configs.dim, 1)
        self.predictor = ConditionedPredictor(dim=configs.dim, num_heads=configs.num_heads, drop_rate=configs.drop_rate,
                                              max_pos_len=configs.max_pos_len, predictor=configs.predictor)


    def forward(self, word_ids, char_ids, video_features, v_mask, q_mask):
        device = video_features.device

        # Project video and query features
        video_features = self.video_proj(video_features)
        video_features = self.feature_encoder(video_features, mask=v_mask)

        query_features = self.embedding_net(word_ids, char_ids)  # (batch_size, seq_len, dim)
        query_features = self.feature_encoder(query_features, mask=q_mask)
        query_features = query_features.mean(dim=1)  # (batch_size, dim)

        # Ensure dimensions match
        query_node = query_features.unsqueeze(1)  # (batch_size, 1, dim)
        all_nodes = torch.cat([query_node, video_features], dim=1)  # (batch_size, num_nodes, dim)

        # Create edge indices for different graphs
        batch_size, num_nodes, hidden_dim = all_nodes.size()
        temporal_edge_index = self.expand_edge_index_for_batch(
            self.build_temporal_edges(video_features.size(1)), batch_size, num_nodes
        ).to(all_nodes.device)
        semantic_edge_index = self.expand_edge_index_for_batch(
            self.build_semantic_edges(video_features.size(1)), batch_size, num_nodes
        ).to(all_nodes.device)
        query_edge_index = self.expand_edge_index_for_batch(
            self.build_query_edges(video_features.size(1)), batch_size, num_nodes
        ).to(all_nodes.device)

        # Flatten for graph processing
        all_nodes_flat = all_nodes.view(-1, hidden_dim)  # (batch_size * num_nodes, hidden_dim)

        # Debugging prints
        #print(f"All Nodes Shape: {all_nodes.shape}")
        #print(f"Temporal Edge Index Shape: {temporal_edge_index.shape}")
        #print(f"Flattened Nodes Shape: {all_nodes_flat.shape}")

        # Process each graph
        temporal_output = all_nodes_flat
        for layer in self.temporal_graph:
            temporal_output = layer(temporal_output, temporal_edge_index)

        semantic_output = all_nodes_flat
        for layer in self.semantic_graph:
            semantic_output = layer(semantic_output, semantic_edge_index)

        query_output = all_nodes_flat
        for layer in self.query_graph:
            query_output = layer(query_output, query_edge_index)

        # Reshape back to batch format
        temporal_output = temporal_output.view(batch_size, num_nodes, -1)
        semantic_output = semantic_output.view(batch_size, num_nodes, -1)
        query_output = query_output.view(batch_size, num_nodes, -1)

        # Concatenate outputs and apply attention
        combined_output = torch.stack([temporal_output, semantic_output, query_output], dim=1)  # (batch_size, 3, num_nodes, dim)
        combined_output = combined_output.mean(dim=1)  # Combine outputs using mean (batch_size, num_nodes, dim)

        # Predict start and end logits
        start_logits = self.start_predictor(combined_output[:, 1:]).squeeze(-1)  # Exclude query node
        end_logits = self.end_predictor(combined_output[:, 1:]).squeeze(-1)  # Exclude query node

        return start_logits, end_logits


    def expand_edge_index_for_batch(self, edge_index, batch_size, num_nodes):
        """
        Expand edge index for batched graph data.
        """
        expanded_edges = []
        for batch_idx in range(batch_size):
            offset = batch_idx * num_nodes
            expanded_edges.append(edge_index + offset)
        return torch.cat(expanded_edges, dim=1)

    def build_temporal_edges(self, num_segments):
        return self.build_graph_edges(num_segments, num_hops=2)

    def build_semantic_edges(self, num_segments):
        # Example: Fully connected graph for semantic similarity
        edges = torch.combinations(torch.arange(num_segments), r=2).t()
        return torch.cat([edges, edges.flip(0)], dim=1)  # Undirected graph

    def build_query_edges(self, num_segments):
        # Connect query node to all video segments
        return torch.tensor([[0] + [i + 1 for i in range(num_segments)],
                            [i + 1 for i in range(num_segments)] + [0]], dtype=torch.long)

    def build_graph_edges(self, num_segments, num_hops=2):
        temporal_edges = torch.tensor([
            [i, i + 1] for i in range(num_segments - 1)
        ] + [
            [i + 1, i] for i in range(num_segments - 1)
        ], dtype=torch.long).t()

        query_edges = torch.tensor([
            [0, i + 1] for i in range(num_segments)
        ] + [
            [i + 1, 0] for i in range(num_segments)
        ], dtype=torch.long).t()

        # Add multi-hop temporal edges
        for hop in range(2, num_hops + 1):
            temporal_edges = torch.cat([
                temporal_edges,
                torch.tensor([
                    [i, i + hop] for i in range(num_segments - hop)
                ] + [
                    [i + hop, i] for i in range(num_segments - hop)
                ], dtype=torch.long).t()
            ], dim=1)

        return torch.cat([temporal_edges, query_edges], dim=1)


    def extract_index(self, start_logits, end_logits):
        return self.predictor.extract_index(start_logits=start_logits, end_logits=end_logits)

    def compute_loss(self, start_logits, end_logits, start_labels, end_labels):
        return self.predictor.compute_cross_entropy_loss(start_logits=start_logits, end_logits=end_logits,
                                                         start_labels=start_labels, end_labels=end_labels)