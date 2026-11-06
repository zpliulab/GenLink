import torch
import torch.nn as nn
from discrete.network_preprocess import PlaceHolder
import math

class DenoisingModel(nn.Module):
    """
    基于Flow Matching的多组学数据图生成模型
    整合基因表达(X)和ATAC数据来引导图的去噪过程
    """

    def __init__(self, num_nodes=99, num_cells=100, hidden_dim=256, num_heads=8, num_layers=4, dropout=0.1, edge_types=2, batch_size=16, num_regions=330):
        super(DenoisingModel, self).__init__()
        self.num_nodes = num_nodes
        self.num_cells = num_cells
        self.edge_types = edge_types
        self.mix_atac = nn.Linear(self.num_cells * 2, self.num_cells)
        self.x_encoder = nn.Sequential(nn.Linear(num_cells, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim))
        self.atac_feature_encoder = nn.Sequential(nn.Linear(num_cells, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim))
        self.g_emb_linear = nn.Linear(1536, hidden_dim)
        self.p_emb_linear = nn.Linear(512 * 2, hidden_dim)
        self.p_a_emb_mix = nn.Linear(hidden_dim * 2, hidden_dim)
        self.x_g_emb_mix = nn.Linear(hidden_dim * 2, hidden_dim)
        self.cross_attn_x2atac = CrossAttentionFusion(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
        self.cross_attn_atac2x = CrossAttentionFusion(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
        self.gate_network = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.fusion_output = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim))
        self.graph_gnn_layers = nn.ModuleList([GraphConvLayer(hidden_dim, hidden_dim, dropout) for _ in range(2)])
        self.time_mlp = nn.Sequential(SinusoidalTimeEmbedding(hidden_dim), nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(), nn.Linear(hidden_dim * 2, hidden_dim), nn.Linear(hidden_dim, 1))
        self.node_pair_proj = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.graph_fusion = HybridGraphFusionBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
        self.output_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, edge_types))
        self.graph_stat_proj = nn.Sequential(nn.Linear(5, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.graph_gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.node_fusion = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))

    def build_atac_reference_graph(self, ATAC, rtf, rg):
        atac_mean = ATAC.mean(axis=-1)
        rtf_w = rtf * atac_mean[:, :, None]
        rg_w = rg * atac_mean[:, :, None]
        atac_matrix = torch.bmm(rtf_w.permute(0, 2, 1), rg)
        atac_mask = (atac_matrix > 1).to(torch.int32)
        return atac_mask

    def fuse_omics_data(self, X, ATAC, rtf, rg, g_emb=None, p_emb=None):
        """高级多组学数据融合"""
        batch_size, num_nodes, num_cells = X.shape
        if p_emb is None:
            atac_gene = torch.einsum('brc, brg -> bgc', ATAC, rg)
            atac_region = torch.einsum('brc, brg -> bgc', ATAC, rtf)
            ATAC = self.mix_atac(torch.cat([atac_gene, atac_region], dim=-1))
            atac_encoded = self.atac_feature_encoder(ATAC)
        else:
            atac_gene_data = torch.einsum('brc, brg -> bgc', ATAC, rg)
            atac_region_data = torch.einsum('brc, brg -> bgc', ATAC, rtf)
            ATAC = self.mix_atac(torch.cat([atac_gene_data, atac_region_data], dim=-1))
            atac_encoded_a = self.atac_feature_encoder(ATAC)
            atac_gene = torch.einsum('brc, brg -> bgc', p_emb, rg)
            atac_region = torch.einsum('brc, brg -> bgc', p_emb, rtf)
            atac_encoded_p = self.p_emb_linear(torch.cat([atac_gene, atac_region], dim=-1))
            atac_encoded = self.p_a_emb_mix(torch.cat([atac_encoded_a, atac_encoded_p], dim=-1))
        if g_emb is None:
            x_encoded = self.x_encoder(X)
        else:
            x_encoded_X = self.x_encoder(X)
            x_encoded_g = self.g_emb_linear(g_emb)
            x_encoded = self.x_g_emb_mix(torch.cat([x_encoded_X, x_encoded_g], dim=-1))
        x_enhanced = self.cross_attn_x2atac(x_encoded, atac_encoded)
        atac_enhanced = self.cross_attn_atac2x(atac_encoded, x_encoded)
        combined = torch.cat([x_enhanced, atac_enhanced], dim=-1)
        gate = self.gate_network(combined)
        gated_fusion = gate * x_enhanced + (1 - gate) * atac_enhanced
        final_concat = torch.cat([gated_fusion, x_enhanced + atac_enhanced], dim=-1)
        fused_features = self.fusion_output(final_concat)
        return fused_features

    def build_multiomics_graph(self, fused_features):
        """基于融合的多组学特征构建图"""
        batch_size, num_nodes, hidden_dim = fused_features.shape
        initial_graph = torch.bmm(fused_features, fused_features.transpose(1, 2)) / hidden_dim ** 0.5
        initial_graph = torch.sigmoid(initial_graph)
        node_features = fused_features
        for gnn_layer in self.graph_gnn_layers:
            node_features = gnn_layer(node_features, initial_graph)
        return node_features

    def adaptive_graph_fusion(self, atac_graph=None, node_feature=None, E=None, t=None, fused_features=None):
        """
        【核心修改】自适应图融合 — 用节点对特征替代 expand
        返回 (batch, nodes, nodes, hidden_dim) 的真正边特征
        """
        E_score = E[..., 1].float()
        t_emb = self.time_mlp(t)
        t_emb = t_emb.unsqueeze(1)
        fused_adj = t_emb * E_score
        A = torch.relu(fused_adj)
        out_mean = A.mean(dim=-1, keepdim=True)
        out_max = A.max(dim=-1, keepdim=True).values
        in_mean = A.mean(dim=1, keepdim=True).transpose(1, 2)
        in_max = A.max(dim=1, keepdim=True).values.transpose(1, 2)
        degree_diff = out_mean - in_mean
        node_graph_stats = torch.cat([out_mean, in_mean, out_max, in_max, degree_diff], dim=-1)
        graph_node_feature = self.graph_stat_proj(node_graph_stats)
        fusion_input = torch.cat([node_feature, graph_node_feature], dim=-1)
        gate = self.graph_gate(fusion_input)
        fusion_delta = self.node_fusion(fusion_input)
        fused_node_feature = node_feature + gate * fusion_delta
        feat_i = fused_node_feature.unsqueeze(2).expand(-1, -1, self.num_nodes, -1)
        feat_j = fused_node_feature.unsqueeze(1).expand(-1, self.num_nodes, -1, -1)
        pair_features = torch.cat([feat_i, feat_j], dim=-1)
        dense_edge_features = self.node_pair_proj(pair_features)
        graph_features = self.graph_fusion(dense_edge_features)
        atac_prior = atac_graph.float()
        atac_prior = (atac_prior - atac_prior.amin(dim=(1, 2), keepdim=True)) / (atac_prior.amax(dim=(1, 2), keepdim=True) - atac_prior.amin(dim=(1, 2), keepdim=True) + 1e-06)
        graph_features = graph_features * (1.0 + atac_prior.unsqueeze(-1))
        return graph_features

    def forward(self, node_mask, noisy_data):
        """前向传播"""
        X = noisy_data['X_t']
        ATAC = noisy_data['ATAC_t']
        E = noisy_data['E_t']
        t = noisy_data['t']
        rtf = noisy_data['rtf']
        rg = noisy_data['rg']
        g_emb = noisy_data['g_emb']
        p_emb = noisy_data['p_emb']
        atac_graph = self.build_atac_reference_graph(ATAC, rtf, rg)
        fused_features = self.fuse_omics_data(X, ATAC, rtf, rg, g_emb, p_emb)
        node_features = self.build_multiomics_graph(fused_features)
        graph_features = self.adaptive_graph_fusion(atac_graph, node_features, E, t)
        E_pred = self.output_proj(graph_features)
        if node_mask is not None:
            edge_mask = node_mask.unsqueeze(2) * node_mask.unsqueeze(1)
            E_pred = E_pred * edge_mask.unsqueeze(-1)
        return PlaceHolder(X=X, ATAC=ATAC, E=E_pred, y=t)

class CrossAttentionFusion(nn.Module):

    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super(CrossAttentionFusion, self).__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 4, hidden_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, query_features, key_value_features):
        attn_out, _ = self.cross_attention(query_features, key_value_features, key_value_features)
        attn_out = self.dropout(attn_out)
        query_features = self.norm1(query_features + attn_out)
        ffn_out = self.ffn(query_features)
        ffn_out = self.dropout(ffn_out)
        enhanced_features = self.norm2(query_features + ffn_out)
        return enhanced_features

class GraphConvLayer(nn.Module):

    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_features, adj_matrix):
        out_degree = adj_matrix.sum(dim=-1, keepdim=True).clamp(min=1.0)
        P = adj_matrix / out_degree
        aggregated = torch.bmm(P.transpose(1, 2), node_features)
        transformed = self.linear(aggregated)
        if node_features.shape[-1] == transformed.shape[-1]:
            transformed = transformed + node_features
        out = self.activation(self.norm(transformed))
        return self.dropout(out)

class SinusoidalTimeEmbedding(nn.Module):
    """正弦位置编码用于时间嵌入"""

    def __init__(self, dim):
        super(SinusoidalTimeEmbedding, self).__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t * embeddings.unsqueeze(0)
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings

class HybridGraphFusionBlock(nn.Module):

    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super(HybridGraphFusionBlock, self).__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.graph_conv = GraphConvLayer(hidden_dim, hidden_dim, dropout)
        self.adj_q = nn.Linear(hidden_dim, hidden_dim)
        self.adj_k = nn.Linear(hidden_dim, hidden_dim)
        self.branch_fusion = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 4, hidden_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph_features):
        """
        Args:
            graph_features: (B, N, N, H)

        Returns:
            output: (B, N, N, H)
        """
        B, N, _, H = graph_features.shape
        graph_flat = graph_features.reshape(B * N, N, H)
        attn_out, _ = self.self_attention(graph_flat, graph_flat, graph_flat)
        attn_out = attn_out.reshape(B, N, N, H)
        gnn_input = graph_flat
        q = self.adj_q(gnn_input)
        k = self.adj_k(gnn_input)
        adj_matrix = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(H)
        adj_matrix = torch.softmax(adj_matrix, dim=-1)
        adj_matrix = self.dropout(adj_matrix)
        gnn_out = self.graph_conv(gnn_input, adj_matrix)
        gnn_out = gnn_out.reshape(B, N, N, H)
        combined = torch.cat([attn_out, gnn_out], dim=-1)
        fused = self.branch_fusion(combined)
        fused = self.dropout(fused)
        graph_features = self.norm1(graph_features + fused)
        ffn_out = self.ffn(graph_features)
        ffn_out = self.dropout(ffn_out)
        output = self.norm2(graph_features + ffn_out)
        return output
