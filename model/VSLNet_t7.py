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
        super().__init__()
        self.configs = configs

        # --------------------------
        # 1) 기존 Embedding & Encoder
        # --------------------------
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

        self.graph_layers = nn.ModuleList([
            TransformerConv(in_channels=configs.dim,
                            out_channels=configs.dim // configs.num_heads,
                            heads=configs.num_heads,
                            edge_dim=configs.dim, 
                            concat=True)
            for _ in range(4)
        ])

        # --------------------------
        # 3) 메타 패스 임베딩 (Edge Attribute 용)
        #    예: num_mp=10
        # --------------------------
        self.num_mp = getattr(configs, 'num_mp', 1080)  # 혹은 10 하드코딩
        self.meta_path_emb = nn.Embedding(self.num_mp, configs.dim)

        # --------------------------
        # 4) Start/End 예측
        # --------------------------
        self.start_predictor = nn.Linear(configs.dim, 1)
        self.end_predictor   = nn.Linear(configs.dim, 1)
        self.predictor = ConditionedPredictor(
            dim=configs.dim, num_heads=configs.num_heads, drop_rate=configs.drop_rate,
            max_pos_len=configs.max_pos_len, predictor=configs.predictor
        )

    def forward(self, word_ids, char_ids, video_features, v_mask, q_mask):
        device = video_features.device

        # ---------------------------------
        # A. 쿼리/비디오 embedding
        # ---------------------------------
        video_features = self.video_proj(video_features)
        video_features = self.feature_encoder(video_features, mask=v_mask)
        query_features = self.embedding_net(word_ids, char_ids)
        query_features = self.feature_encoder(query_features, mask=q_mask)
        query_features = query_features.mean(dim=1)  # (B, dim)

        # ---------------------------------
        # B. 노드 배치
        # ---------------------------------
        B, num_segments, dim = video_features.shape
        query_node  = query_features.unsqueeze(1)          # (B,1,dim)
        all_nodes   = torch.cat([query_node, video_features], dim=1)
        # => 노드 인덱스:
        #    idx=0: query
        #    idx=1..N: video

        total_nodes = 1 + num_segments

        # ---------------------------------
        # C. 그래프 엣지/엣지 속성(edge_attr) 구성
        # ---------------------------------
        # build_graph_edges_with_attr() -> returns (edge_index, edge_attr_ids)
        edge_index, edge_attr_ids = self.build_graph_edges_with_attr(num_segments)
        edge_index  = edge_index.to(device)
        edge_attr_ids = edge_attr_ids.to(device)  # shape=[E]

        # edge_attr_ids: 각 엣지의 meta-path ID (0..num_mp-1)
        # => 임베딩화
        edge_attr_emb = self.meta_path_emb(edge_attr_ids)  # shape=[E, dim]

        # ---------------------------------
        # D. batch를 위해 Flatten
        # ---------------------------------
        all_nodes = all_nodes.view(B * total_nodes, dim)
        edge_index = self.expand_edge_index_for_batch(edge_index, B, total_nodes)

        # edge_attr_emb도 batch_size 만큼 복제 (edge별로)
        # => 방법 1) batch 내 모든 그래프에서 동일 패턴 ID
        #    간단히 repeat
        E = edge_attr_emb.size(0)
        edge_attr_emb = edge_attr_emb.unsqueeze(0).expand(B, E, dim)
        # shape = (B,E,dim)
        edge_attr_emb = edge_attr_emb.contiguous().view(-1, dim)
        # shape = (B*E, dim)

        # ---------------------------------
        # E. Graph Transformer Layers
        # ---------------------------------
        x = all_nodes
        for layer in self.graph_layers:
            x = layer((x, x), edge_index, edge_attr=edge_attr_emb)
            # => PyG에서 (x, x) == (node_feats_src, node_feats_dst)

        # ---------------------------------
        # F. 다시 (B, total_nodes, dim)로
        # ---------------------------------
        x = x.view(B, total_nodes, dim)

        # ---------------------------------
        # G. Start/End 예측 (video 노드만)
        # ---------------------------------
        # video 노드: idx=1..N
        video_feats = x[:, 1:1+num_segments, :]
        start_logits = self.start_predictor(video_feats).squeeze(-1)
        end_logits   = self.end_predictor(video_feats).squeeze(-1)

        return start_logits, end_logits

    # ----------------------------------------------------------------
    # build_graph_edges_with_attr:
    #  예) query (idx=0), video (1..N)
    #  -> query->video edge마다 meta_path id를 0번,
    #     video->video edge마다 meta_path id를 1번
    #     ... 식으로 임의로 지정(예시).
    # ----------------------------------------------------------------
    def build_graph_edges_with_attr(self, num_segments):
        """
        Returns:
          edge_index: shape=[2, E]
          edge_attr_ids: shape=[E], each in [0..self.num_mp-1]
        """
        edges = []
        edge_attrs = []

        for v in range(1, num_segments+1):
            edges.append([0, v])  # query->video
            edge_attrs.append(0)
            edges.append([v, 0])  # video->query
            edge_attrs.append(0)

        # [Case2] Video <-> Video (temporal) -> meta_path_id=1
        for i in range(1, num_segments):
            edges.append([i, i+1])
            edge_attrs.append(1)
            edges.append([i+1, i])
            edge_attrs.append(1)

        # (추가적인 edge가 필요하다면 더 작성)
        # 예) multi-hop edge -> meta_path_id=2
        # 예) 나중에는 랜덤하게 meta_path_id 배정할 수도 있음.

        edges      = torch.tensor(edges, dtype=torch.long)      # shape=[E, 2]
        edge_index = edges.t().contiguous()                      # shape=[2, E]
        edge_attr_ids = torch.tensor(edge_attrs, dtype=torch.long)  # shape=[E]

        return edge_index, edge_attr_ids

    def expand_edge_index_for_batch(self, edge_index, batch_size, num_nodes):
        # edge_index: [2, E]
        all_edges = []
        for b in range(batch_size):
            offset = b * num_nodes
            all_edges.append(edge_index + offset)
        all_edges = torch.cat(all_edges, dim=1)  # [2, E * B]
        return all_edges

    # ----------------------------------------------------------------
    # (Optional) Loss & predict
    # ----------------------------------------------------------------
    def compute_loss(self, start_logits, end_logits, start_labels, end_labels):
        return self.predictor.compute_cross_entropy_loss(
            start_logits=start_logits,
            end_logits=end_logits,
            start_labels=start_labels,
            end_labels=end_labels
        )
    
    def extract_index(self, start_logits, end_logits):
        return self.predictor.extract_index(start_logits, end_logits)
