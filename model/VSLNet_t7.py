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
from torch_geometric.nn import TransformerConv

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
        self.feature_encoder = FeatureEncoder(
            dim=configs.dim, num_heads=configs.num_heads, kernel_size=7, num_layers=4,
            max_pos_len=configs.max_pos_len, drop_rate=configs.drop_rate
        )

        # Unified Graph Layer
        self.shared_graph_layer = TransformerConv(
            configs.dim, configs.dim // configs.num_heads, heads=configs.num_heads
        )

        # Start-End Time Prediction
        self.start_predictor = nn.Linear(configs.dim, 1)
        self.end_predictor = nn.Linear(configs.dim, 1)
        self.predictor = ConditionedPredictor(
            dim=configs.dim, num_heads=configs.num_heads, drop_rate=configs.drop_rate,
            max_pos_len=configs.max_pos_len, predictor=configs.predictor
        )

    def forward(self, word_ids, char_ids, video_features, v_mask, q_mask):
        device = video_features.device

        # Project video and query features
        video_features = self.video_proj(video_features).to(device)
        #print(f"[DEBUG] Video Features Shape: {video_features.shape}, Device: {video_features.device}")
        
        video_features = self.feature_encoder(video_features, mask=v_mask)
        #print(f"[DEBUG] Encoded Video Features Shape: {video_features.shape}, Device: {video_features.device}")

        query_features = self.embedding_net(word_ids, char_ids).to(device)
        #print(f"[DEBUG] Query Features Shape: {query_features.shape}, Device: {query_features.device}")
        
        query_features = self.feature_encoder(query_features, mask=q_mask)
        query_features = query_features.mean(dim=1).to(device)
        #print(f"[DEBUG] Pooled Query Features Shape: {query_features.shape}, Device: {query_features.device}")

        # Ensure dimensions match
        query_node = query_features.unsqueeze(1).to(device)
        #print(f"[DEBUG] Query Node Shape: {query_node.shape}, Device: {query_node.device}")
        
        all_nodes = torch.cat([query_node, video_features], dim=1).to(device)
        #print(f"[DEBUG] All Nodes Shape: {all_nodes.shape}, Device: {all_nodes.device}")

        # Create edge indices for different graphs
        batch_size, num_nodes, hidden_dim = all_nodes.size()
        temporal_edge_index = self.expand_edge_index_for_batch(
            self.build_temporal_edges(video_features.size(1)), batch_size, num_nodes
        ).to(device)
        #print(f"[DEBUG] Temporal Edge Index Shape: {temporal_edge_index.shape}, Device: {temporal_edge_index.device}")

        semantic_edge_index = self.expand_edge_index_for_batch(
            self.build_semantic_edges(video_features.size(1)), batch_size, num_nodes
        ).to(device)
        #print(f"[DEBUG] Semantic Edge Index Shape: {semantic_edge_index.shape}, Device: {semantic_edge_index.device}")

        query_edge_index = self.expand_edge_index_for_batch(
            self.build_query_edges(video_features.size(1)), batch_size, num_nodes
        ).to(device)

        combined_edge_index = torch.cat([temporal_edge_index, semantic_edge_index, query_edge_index], dim=1).to(device)

        all_nodes_flat = all_nodes.view(-1, hidden_dim).to(device)
        combined_output = self.shared_graph_layer(all_nodes_flat, combined_edge_index)

        combined_output = combined_output.view(batch_size, num_nodes, -1)

        start_logits = self.start_predictor(combined_output[:, 1:]).squeeze(-1)
        end_logits = self.end_predictor(combined_output[:, 1:]).squeeze(-1)

        return start_logits, end_logits

    def subsample_nodes(self, features, stride):
        """Subsample nodes by a given stride to reduce memory usage."""
        return features[:, ::stride, :]

    def build_temporal_edges(self, num_segments):
        return self.build_graph_edges(num_segments, num_hops=2)

    def build_semantic_edges(self, num_segments):
        edges = torch.combinations(torch.arange(num_segments), r=2).t()
        return torch.cat([edges, edges.flip(0)], dim=1)

    def build_query_edges(self, num_segments):
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

    def expand_edge_index_for_batch(self, edge_index, batch_size, num_nodes):
        """Expand edge index for batched graph data."""
        expanded_edges = []
        for batch_idx in range(batch_size):
            offset = batch_idx * num_nodes
            expanded_edges.append(edge_index + offset)
        return torch.cat(expanded_edges, dim=1)

    def extract_index(self, start_logits, end_logits):
        return self.predictor.extract_index(start_logits=start_logits, end_logits=end_logits)

    def compute_loss(self, start_logits, end_logits, start_labels, end_labels):
        return self.predictor.compute_cross_entropy_loss(start_logits=start_logits, end_logits=end_logits,
                                                         start_labels=start_labels, end_labels=end_labels)
