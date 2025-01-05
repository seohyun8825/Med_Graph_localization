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
from torch_geometric.data import Data

from model.layers_t7 import Embedding, VisualProjection, FeatureEncoder, ConditionedPredictor
from transformers import AdamW, get_linear_schedule_with_warmup

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
            dim=configs.dim, num_heads=configs.num_heads, kernel_size=7,
            num_layers=4, max_pos_len=configs.max_pos_len, drop_rate=configs.drop_rate
        )

        # Graph Transformer Layers
        self.graph_layers = nn.ModuleList([
            TransformerConv(configs.dim, configs.dim // configs.num_heads, heads=configs.num_heads)
            for _ in range(4)
        ])

        # 예: num_mp=10, "중간 노드" 10개
        self.num_mp = 10
        self.meta_path_emb = nn.Embedding(self.num_mp, configs.dim)

        # Start-End Time Prediction
        self.start_predictor = nn.Linear(configs.dim, 1)
        self.end_predictor   = nn.Linear(configs.dim, 1)
        self.predictor = ConditionedPredictor(
            dim=configs.dim, num_heads=configs.num_heads, drop_rate=configs.drop_rate,
            max_pos_len=configs.max_pos_len, predictor=configs.predictor
        )

    def forward(self, word_ids, char_ids, video_features, v_mask, q_mask):
        device = video_features.device

        # 1) Video / Query 임베딩
        video_features = self.video_proj(video_features)
        video_features = self.feature_encoder(video_features, mask=v_mask)
        query_features = self.embedding_net(word_ids, char_ids)
        query_features = self.feature_encoder(query_features, mask=q_mask)
        query_features = query_features.mean(dim=1)  # (B, dim)

        # 노드 쌓기
        B, num_segments, dim = video_features.shape
        # a) Query node
        query_node = query_features.unsqueeze(1)  # (B, 1, dim)

        # b) Meta-Path nodes (M=10)
        #    => (B, 10, dim)
        mp_index  = torch.arange(self.num_mp, device=device).unsqueeze(0).repeat(B,1)  # (B,10)
        mp_nodes  = self.meta_path_emb(mp_index)  # (B, 10, dim)

        # c) Video nodes => shape (B, N, dim)

        # 최종 노드 순서:
        #   idx=0                : query
        #   idx=1..N            : video
        #   idx=N+1..N+10       : meta path
        all_nodes = torch.cat([query_node, video_features, mp_nodes], dim=1)
        # shape = (B, 1 + N + 10, dim)

        # 2) 그래프 엣지 구성
        edge_index = self.build_graph_edges(num_segments, self.num_mp).to(device)

        # 3) Flatten for PyG
        batch_size, total_nodes, hidden_dim = all_nodes.size()
        all_nodes = all_nodes.view(batch_size * total_nodes, hidden_dim)

        # 4) Expand edge index for batch
        edge_index = self.expand_edge_index_for_batch(edge_index, batch_size, total_nodes)

        # 5) Graph Transformer
        for layer in self.graph_layers:
            all_nodes = layer(all_nodes, edge_index)

        # 6) Reshape back
        all_nodes = all_nodes.view(batch_size, total_nodes, hidden_dim)

        # 7) Start/End logits for video nodes
        #    video = idx [1..N]
        start_logits = self.start_predictor(all_nodes[:, 1:1+num_segments]).squeeze(-1)
        end_logits   = self.end_predictor(all_nodes[:, 1:1+num_segments]).squeeze(-1)

        return start_logits, end_logits

    def build_graph_edges(self, num_segments, num_mp):
        """
        노드 인덱스:
          0            : query
          1..N         : video
          N+1..N+M     : meta path nodes
        """
        edges = []
        mp_start = num_segments + 1
        mp_end   = num_segments + num_mp  # inclusive index

        # 1) Query <-> ALL MP nodes
        for mp_i in range(mp_start, mp_end+1):
            edges.append([0, mp_i])   # query->mp
            edges.append([mp_i, 0])   # mp->query

        # 2) MP node <-> ALL Video
        for mp_i in range(mp_start, mp_end+1):
            for v in range(1, num_segments+1):
                edges.append([mp_i, v])   # mp->video
                edges.append([v, mp_i])   # video->mp

        # 3) (Optional) Video <-> Video (temporal)
        for i in range(1, num_segments):
            edges.append([i, i+1])
            edges.append([i+1, i])

        # 4) (Optional) Query 노드 <-> Video 노드 (직접 연결)
        for v in range(1, num_segments + 1):
            edges.append([0, v])
            edges.append([v, 0])

        edge_index = torch.tensor(edges, dtype=torch.long).t()  # shape=[2, E]
        return edge_index

    def expand_edge_index_for_batch(self, edge_index, batch_size, num_nodes):
        expanded_edges = []
        for b in range(batch_size):
            offset = b * num_nodes
            expanded_edges.append(edge_index + offset)
        expanded_edges = torch.cat(expanded_edges, dim=1)
        return expanded_edges

    def compute_loss(self, start_logits, end_logits, start_labels, end_labels):
        return self.predictor.compute_cross_entropy_loss(
            start_logits=start_logits, 
            end_logits=end_logits,
            start_labels=start_labels, 
            end_labels=end_labels
        )

    def extract_index(self, start_logits, end_logits):
        return self.predictor.extract_index(start_logits, end_logits)
