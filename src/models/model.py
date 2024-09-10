import math

import torch
from torch.nn import ModuleList, Linear, Conv1d, MaxPool1d, Embedding, BatchNorm1d, BatchNorm2d
import torch.nn.functional as F
from torch_geometric.nn import aggr, GCNConv

from src.models.gps_conv import GPSConv
from src.models.vit import ViT
from src.models.pair_aggr import PairAggregation


class TopoLink(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, max_z=1000, k=0.6, train_dataset=None,
                 dynamic_train=False, GNN=GCNConv, use_feature=False, use_node_label=True,
                 node_embedding=None, gnn_dropout=0, attn_dropout=0, mlp_dropout=0,
                 attn_type='sga', num_heads=1, use_ph=True, pi_dim=50, patch_size=5,
                 vit_dim=128, vit_depth=2, vit_out_dim=256, vit_headers=4, vit_mlp_dim=128,
                 use_pe=False, walk_length=32, out_mlp_dim=128):
        super(TopoLink, self).__init__()

        self.use_feature = use_feature
        self.node_embedding = node_embedding
        self.mlp_dropout = mlp_dropout
        self.use_pe = use_pe
        self.use_node_label = use_node_label

        if k <= 1:  # Transform percentile to number.
            if train_dataset is None:
                k = 30
            else:
                if dynamic_train:
                    sampled_train = train_dataset[:1000]
                else:
                    sampled_train = train_dataset
                num_nodes = sorted([g.num_nodes for g in sampled_train])
                k = num_nodes[int(math.ceil(k * len(num_nodes))) - 1]
                k = max(10, k)
        self.k = int(k)

        if use_node_label:
            self.max_z = max_z
            self.z_embedding = Embedding(self.max_z, hidden_channels)

        initial_channels = 0
        if use_node_label:
            initial_channels += hidden_channels
        if self.use_feature:
            initial_channels += train_dataset.num_features
        if self.node_embedding is not None:
            initial_channels += node_embedding.embedding_dim
        if self.use_pe:
            initial_channels += walk_length
            self.pe_norm = BatchNorm1d(walk_length)
        assert initial_channels > 0

        self.lin_in = Linear(initial_channels, hidden_channels)
        self.convs = ModuleList()
        for _ in range(num_layers):
            conv = GPSConv(hidden_channels, GNN(hidden_channels, hidden_channels),
                           heads=num_heads, dropout=gnn_dropout,
                           attn_type=attn_type, attn_kwargs={'dropout': attn_dropout})
            self.convs.append(conv)

        self.pool = aggr.SortAggregation(self.k)
        conv1d_channels = [16, 32]
        conv1d_kws = [hidden_channels, 5]
        self.conv1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0], conv1d_kws[0])
        self.maxpool1d = MaxPool1d(2, 2)
        self.conv2 = Conv1d(conv1d_channels[0], conv1d_channels[1], conv1d_kws[1], 1)
        dense_dim = int((self.k - 2) / 2 + 1)
        dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]

        self.use_ph = use_ph
        if use_ph:
            self.pi_norm = BatchNorm2d(1)
            self.vit = ViT(
                image_size=(pi_dim, pi_dim),
                patch_size=patch_size,
                out_dim=vit_out_dim,
                dim=vit_dim,
                depth=vit_depth,
                heads=vit_headers,
                mlp_dim=vit_mlp_dim,
                dropout=0.0,
                emb_dropout=0.0
            )
            dense_dim += vit_out_dim

        self.pair_aggr = PairAggregation(hidden_channels)
        dense_dim += hidden_channels

        self.lin1 = Linear(dense_dim, out_mlp_dim)
        self.lin2 = Linear(out_mlp_dim, 1)

    def forward(self, z, edge_index, batch, feat=None, edge_weight=None, node_id=None, pi=None, pe=None):
        x = None
        if self.use_node_label:
            x = self.z_embedding(z)
            if x.ndim == 3:  # in case z has multiple integer labels
                x = x.sum(dim=1)
        if self.use_feature and feat is not None:
            if x is None:
                x = feat.to(torch.float)
            else:
                x = torch.cat([x, feat.to(torch.float)], 1)
        if self.node_embedding is not None and node_id is not None:
            n_emb = self.node_embedding(node_id)
            if x is None:
                x = n_emb
            else:
                x = torch.cat([x, n_emb], 1)
        if self.use_pe and pe is not None:
            pe = self.pe_norm(pe)
            if x is None:
                x = pe
            else:
                x = torch.cat([x, pe], 1)
        assert x is not None
        x = self.lin_in(x)

        # GraphGPS.
        for conv in self.convs:
            x = conv(x, edge_index, batch, edge_weight=edge_weight)
        h = x

        # Global pooling.
        x = self.pool(x, batch)
        x = x.unsqueeze(1)  # [num_graphs, 1, k * hidden]
        x = F.relu(self.conv1(x))
        x = self.maxpool1d(x)
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # [num_graphs, dense_dim]

        # ViT for persistent images.
        if self.use_ph and pi is not None:
            batch_size = int(batch.max()) + 1
            pi = pi.view((batch_size, 1, -1, pi.size(-1)))  # [num_graphs, 1, pi_dim, pi_dim]
            pi = self.pi_norm(pi)
            pi = self.vit(pi)   # [num_graphs, mixer_out_dim]
            x = torch.cat([x, pi], 1)

        # Pairwise product of target node pair.
        h = self.pair_aggr(h, batch)
        x = torch.cat([x, h], 1)

        # MLP.
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.mlp_dropout, training=self.training)
        x = self.lin2(x)
        return x
