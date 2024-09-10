from typing import Any, Optional

import networkx as nx
import numpy as np
import gudhi as gd
import torch

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


def persistence_diagram(A, method, max_scale=50, extend=False):
    G = nx.from_numpy_array(A)
    if method == 'degree' or method == 'ricci':
        node_features = np.sum(A, axis=1)
    elif method == 'betweenness':
        node_features_dict = nx.betweenness_centrality(G, weight='weight')
        node_features = [i for i in node_features_dict.values()]
    elif method == 'communicability':
        node_features_dict = nx.communicability_betweenness_centrality(G)
        node_features = [i for i in node_features_dict.values()]
    elif method == 'eigenvector':
        node_features_dict = nx.eigenvector_centrality(G, max_iter=10000, weight='weight')
        node_features = [i for i in node_features_dict.values()]
    elif method == 'closeness':
        node_features_dict = nx.closeness_centrality(G, distance='weight')
        node_features = [i for i in node_features_dict.values()]
    else:
        raise ValueError('Not support %s.' % method)

    stb = gd.SimplexTree()
    (xs, ys) = np.where(np.triu(A))
    for j in range(A.shape[0]):
        stb.insert([j], filtration=-1e10)

    for idx, x in enumerate(xs):
        stb.insert([x, ys[idx]], filtration=-1e10)

    for j in range(A.shape[0]):
        stb.assign_filtration([j], node_features[j])

    stb.make_filtration_non_decreasing()
    if extend:  # use extended persistence
        stb.extend_filtration()
        dgm = stb.extended_persistence()
        pd = [dgm[0][i][1] for i in range(len(dgm[0]))] + \
             [dgm[1][i][1] for i in range(len(dgm[1]))] + \
             [dgm[2][i][1] for i in range(len(dgm[2]))] + \
             [dgm[3][i][1] for i in range(len(dgm[3]))]
        if len(pd) == 0:
            pd = [(0, max_scale)]
    else:   # use ordinary persistence
        dgm = stb.persistence()
        pd = [dgm[i][1] if dgm[i][1][1] != np.inf else (dgm[i][1][0], max_scale)
              for i in range(len(dgm))]

    return np.array(pd)


def persistence_images(dgm, resolution=(50, 50), return_raw=False, normalization=True, bandwidth=1., power=1.):
    PXs, PYs = dgm[:, 0], dgm[:, 1]
    xm, xM, ym, yM = PXs.min(), PXs.max(), PYs.min(), PYs.max()
    x = np.linspace(xm, xM, resolution[0])
    y = np.linspace(ym, yM, resolution[1])
    X, Y = np.meshgrid(x, y)
    X, Y = X[:, :, np.newaxis], Y[:, :, np.newaxis]

    P0, P1 = np.reshape(dgm[:, 0], [1, 1, -1]), np.reshape(dgm[:, 1], [1, 1, -1])
    weight = np.abs(P1 - P0)
    distpts = np.sqrt((X - P0) ** 2 + (Y - P1) ** 2)

    if return_raw:
        lw = [weight[0, 0, pt] for pt in range(weight.shape[2])]
        lsum = [distpts[:, :, pt] for pt in range(distpts.shape[2])]
        output = [lw, lsum]
    else:
        weight = weight ** power
        Zfinal = (np.multiply(weight, np.exp(-distpts ** 2 / bandwidth))).sum(axis=2)
        output = Zfinal

    max_output = np.max(output)
    min_output = np.min(output)
    if normalization and (max_output != min_output):
        norm_output = (output - min_output)/(max_output - min_output)
    else:
        norm_output = output

    return norm_output


def add_node_attr(
    data: Data,
    value: Any,
    attr_name: Optional[str] = None,
) -> Data:
    if attr_name is None:
        if data.x is not None:
            x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([x, value.to(x.device, x.dtype)], dim=-1)
        else:
            data.x = value
    else:
        data[attr_name] = value

    return data


@functional_transform('add_persistence_images')
class AddPersistenceImages(BaseTransform):
    def __init__(
        self,
        pi_dim: int,
        filtration_method: str = 'degree',
        extend: bool = False,
        attr_name: Optional[str] = 'pi'
    ) -> None:
        self.pi_dim = pi_dim
        self.filtration_method = filtration_method
        self.extend = extend
        self.attr_name = attr_name

    def forward(self, data: Data) -> Data:
        if self.filtration_method == 'ricci':
            adj = data.ricci.toarray()
        else:
            row, col = data.edge_index
            N = data.num_nodes
            adj = torch.zeros((N, N), device=row.device)
            adj[row, col] = 1
            adj = adj.numpy()

        pd = persistence_diagram(adj, self.filtration_method, extend=self.extend)
        pi = persistence_images(pd, resolution=(self.pi_dim, self.pi_dim))
        pi = torch.FloatTensor(pi)

        data = add_node_attr(data, pi, attr_name=self.attr_name)
        return data
