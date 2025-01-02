import torch
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt

def visualize_graph(edge_index, num_nodes, temporal_edges, semantic_edges, query_edges, query_node_idx=0):


    graph = nx.Graph()

    # Temporal 엣지 추가
    temporal_edges_np = temporal_edges.cpu().numpy()
    graph.add_edges_from(temporal_edges_np.T)

    # Semantic 엣지 추가
    semantic_edges_np = semantic_edges.cpu().numpy()
    graph.add_edges_from(semantic_edges_np.T)

    # Query 엣지 추가
    query_edges_np = query_edges.cpu().numpy()
    graph.add_edges_from(query_edges_np.T)

    # 실제 그래프의 노드 목록 얻기
    graph_nodes = list(graph.nodes)

    # 노드 색상 설정 (쿼리 노드와 비디오 피처 노드 구분)
    node_colors = ['red' if i == query_node_idx else 'blue' for i in graph_nodes]

    # 엣지 색상 설정
    edge_colors = []
    for edge in graph.edges:
        if edge in list(zip(temporal_edges_np[0], temporal_edges_np[1])):
            edge_colors.append('green')  # Temporal 엣지
        elif edge in list(zip(semantic_edges_np[0], semantic_edges_np[1])):
            edge_colors.append('orange')  # Semantic 엣지
        elif edge in list(zip(query_edges_np[0], query_edges_np[1])):
            edge_colors.append('purple')  # Query 엣지
        else:
            edge_colors.append('black')  # 기타 엣지

    # 그래프 시각화
    plt.figure(figsize=(10, 6))
    nx.draw(
        graph,
        node_color=node_colors,
        edge_color=edge_colors,
        with_labels=True,
        font_weight='bold',
        node_size=500,
        font_size=10
    )
    plt.title("Graph Visualization with Different Edge Types")
    plt.show()

# 예시 데이터 생성
num_video_nodes = 5  # 비디오 피처 노드의 개수
query_node_idx = 0   # 쿼리 노드의 인덱스
num_nodes = num_video_nodes + 1  # 쿼리 노드를 포함한 총 노드 수

# Temporal, Semantic, Query 엣지 정의
temporal_edges = torch.tensor([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]], dtype=torch.long)  # 예시
semantic_edges = torch.tensor([[i, j] for i in range(1, 6) for j in range(1, 6) if i != j], dtype=torch.long).T
query_edges = torch.tensor([[query_node_idx] * num_video_nodes, list(range(1, num_video_nodes + 1))], dtype=torch.long)

# 전체 엣지 통합
combined_edges = torch.cat([temporal_edges, semantic_edges, query_edges], dim=1)

# 그래프 시각화
visualize_graph(combined_edges, num_nodes, temporal_edges, semantic_edges, query_edges)
