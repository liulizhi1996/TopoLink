from math import inf
import random
import os.path as osp

import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.utils import (negative_sampling, add_self_loops)
import torch_geometric.transforms as T
from torch_sparse import coalesce

import numpy as np
from tqdm import tqdm
import scipy.sparse as ssp
import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci

from src.models.labelling_tricks import drnl_node_labeling, de_node_labeling, de_plus_node_labeling
from src.utils.tda_utils import AddPersistenceImages


class SEALDataset(InMemoryDataset):
    def __init__(self, root, data, pos_edges, neg_edges, num_hops, percent=1., split='train',
                 use_coalesce=False, node_label='drnl', ratio_per_hop=1.0,
                 max_nodes_per_hop=None, max_dist=1000, directed=False, sign=False, k=None,
                 pi_dim=50, filtration='degree', extend=False, walk_length=32):
        self.data = data
        self.pos_edges = pos_edges
        self.neg_edges = neg_edges
        self.num_hops = num_hops
        self.percent = int(percent) if percent >= 1.0 else percent
        self.split = split
        self.use_coalesce = use_coalesce
        self.node_label = node_label
        self.ratio_per_hop = ratio_per_hop
        self.max_nodes_per_hop = max_nodes_per_hop
        self.max_dist = max_dist
        self.directed = directed
        self.sign = sign
        self.k = k
        self.pi_dim = pi_dim
        self.filtration = filtration
        self.extend = extend
        self.walk_length = walk_length
        super(SEALDataset, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        if self.percent == 1.:
            name = f'SEAL_{self.split}_data'
        else:
            name = f'SEAL_{self.split}_data_{self.percent}'
        name += '.pt'
        return [name]

    def process(self):
        if self.use_coalesce:  # compress multi-edge into edge with weight
            self.data.edge_index, self.data.edge_weight = coalesce(
                self.data.edge_index, self.data.edge_weight,
                self.data.num_nodes, self.data.num_nodes)

        if 'edge_weight' in self.data:
            edge_weight = self.data.edge_weight.view(-1)
        else:
            edge_weight = torch.ones(self.data.edge_index.size(1), dtype=int)
        A = ssp.csr_matrix(
            (edge_weight, (self.data.edge_index[0], self.data.edge_index[1])),
            shape=(self.data.num_nodes, self.data.num_nodes)
        )

        if self.directed:
            A_csc = A.tocsc()
        else:
            A_csc = None

        if self.filtration == 'ricci':
            ricci = compute_ricci_curvature(self.data)
        else:
            ricci = None

        # Extract enclosing subgraphs for pos and neg edges
        pos_list = extract_enclosing_subgraphs(
            self.pos_edges, A, self.data.x, 1, self.num_hops, self.node_label,
            self.ratio_per_hop, self.max_nodes_per_hop, self.max_dist, self.directed, A_csc,
            pi_dim=self.pi_dim, filtration_method=self.filtration, extend=self.extend,
            walk_length=self.walk_length, ricci=ricci)
        neg_list = extract_enclosing_subgraphs(
            self.neg_edges, A, self.data.x, 0, self.num_hops, self.node_label,
            self.ratio_per_hop, self.max_nodes_per_hop, self.max_dist, self.directed, A_csc,
            pi_dim=self.pi_dim, filtration_method=self.filtration, extend=self.extend,
            walk_length=self.walk_length, ricci=ricci)

        torch.save(self.collate(pos_list + neg_list), self.processed_paths[0])
        del pos_list, neg_list


class SEALDynamicDataset(Dataset):
    def __init__(self, root, data, pos_edges, neg_edges, num_hops, percent=1., use_coalesce=False, node_label='drnl',
                 ratio_per_hop=1.0, max_nodes_per_hop=None, max_dist=1000, directed=False, sign=False, k=None,
                 pi_dim=50, filtration='degree', extend=False, walk_length=32, **kwargs):
        self.data = data
        self.pos_edges = pos_edges
        self.neg_edges = neg_edges
        self.num_hops = num_hops
        self.percent = percent
        self.use_coalesce = use_coalesce
        self.node_label = node_label
        self.ratio_per_hop = ratio_per_hop
        self.max_nodes_per_hop = max_nodes_per_hop
        self.max_dist = max_dist
        self.directed = directed
        self.sign = sign
        self.k = k
        self.pi_dim = pi_dim
        self.filtration = filtration
        self.extend = extend
        self.walk_length = walk_length
        super(SEALDynamicDataset, self).__init__(root)

        self.links = torch.cat([self.pos_edges, self.neg_edges], 0).tolist()
        self.labels = [1] * self.pos_edges.size(0) + [0] * self.neg_edges.size(0)

        if self.use_coalesce:  # compress mutli-edge into edge with weight
            self.data.edge_index, self.data.edge_weight = coalesce(
                self.data.edge_index, self.data.edge_weight,
                self.data.num_nodes, self.data.num_nodes)

        if 'edge_weight' in self.data:
            edge_weight = self.data.edge_weight.view(-1)
        else:
            edge_weight = torch.ones(self.data.edge_index.size(1), dtype=int)
        self.A = ssp.csr_matrix(
            (edge_weight, (self.data.edge_index[0], self.data.edge_index[1])),
            shape=(self.data.num_nodes, self.data.num_nodes)
        )
        if self.directed:
            self.A_csc = self.A.tocsc()
        else:
            self.A_csc = None
        if self.filtration == 'ricci':
            self.ricci = compute_ricci_curvature(self.data)
        else:
            self.ricci = None

    def len(self):
        return len(self.links)

    def get(self, idx):
        src, dst = self.links[idx]
        y = self.labels[idx]
        src_degree, dst_degree = get_src_dst_degree(src, dst, self.A, self.max_nodes_per_hop)
        if self.sign:
            x = [self.data.x]
            x += [self.data[f'x{i}'] for i in range(1, self.k + 1)]
        else:
            x = self.data.x
        tmp = k_hop_subgraph(src, dst, self.num_hops, self.A, self.ratio_per_hop,
                             self.max_nodes_per_hop, node_features=x,
                             y=y, directed=self.directed, A_csc=self.A_csc, ricci=self.ricci)
        data = construct_pyg_graph(*tmp, self.node_label, self.max_dist, src_degree, dst_degree)
        transform = T.Compose([
            AddPersistenceImages(pi_dim=self.pi_dim, filtration_method=self.filtration, extend=self.extend),
            T.AddRandomWalkPE(walk_length=self.walk_length, attr_name='pe')
        ])
        data = transform(data)

        return data


def sample_data(data, sample_arg):
    if sample_arg <= 1:
        samples = int(sample_arg * len(data))
    elif sample_arg != inf:
        samples = int(sample_arg)
    else:
        samples = len(data)
    if samples != inf:
        sample_indices = torch.randperm(len(data))[:samples]
    return data[sample_indices]


def get_train_val_test_datasets(dataset, train_data, val_data, test_data, args):
    path = osp.join(dataset.root, dataset.name)
    use_coalesce = True if args.dataset_name == 'ogbl-collab' else False
    # get percents used only for naming the SEAL dataset files and caching
    train_percent, val_percent, test_percent = 1 - (args.val_pct + args.test_pct), args.val_pct, args.test_pct
    # probably should be an attribute of the dataset and not hardcoded
    directed = False
    pos_train_edge, neg_train_edge = get_pos_neg_edges(train_data)
    pos_val_edge, neg_val_edge = get_pos_neg_edges(val_data)
    pos_test_edge, neg_test_edge = get_pos_neg_edges(test_data)
    print(
        f'before sampling, considering a superset of {pos_train_edge.shape[0]} pos, {neg_train_edge.shape[0]} neg train edges '
        f'{pos_val_edge.shape[0]} pos, {neg_val_edge.shape[0]} neg val edges '
        f'and {pos_test_edge.shape[0]} pos, {neg_test_edge.shape[0]} neg test edges for supervision')

    pos_train_edge = sample_data(pos_train_edge, args.train_samples)
    neg_train_edge = sample_data(neg_train_edge, args.train_samples)
    pos_val_edge = sample_data(pos_val_edge, args.val_samples)
    neg_val_edge = sample_data(neg_val_edge, args.val_samples)
    pos_test_edge = sample_data(pos_test_edge, args.test_samples)
    neg_test_edge = sample_data(neg_test_edge, args.test_samples)

    print(
        f'after sampling, using {pos_train_edge.shape[0]} pos, {neg_train_edge.shape[0]} neg train edges '
        f'{pos_val_edge.shape[0]} pos, {neg_val_edge.shape[0]} neg val edges '
        f'and {pos_test_edge.shape[0]} pos, {neg_test_edge.shape[0]} neg test edges for supervision')

    dataset_class = 'SEALDynamicDataset' if args.dynamic_train else 'SEALDataset'
    train_dataset = eval(dataset_class)(
        path,
        train_data,
        pos_train_edge,
        neg_train_edge,
        num_hops=args.num_hops,
        percent=train_percent,
        split='train',
        use_coalesce=use_coalesce,
        node_label=args.node_label,
        ratio_per_hop=args.ratio_per_hop,
        max_nodes_per_hop=args.max_nodes_per_hop,
        max_dist=args.max_dist,
        directed=directed,
        sign=args.sign_k > 0,
        k=args.sign_k,
        pi_dim=args.pi_dim,
        filtration=args.filtration,
        extend=args.extend,
        walk_length=args.walk_length
    )
    dataset_class = 'SEALDynamicDataset' if args.dynamic_val else 'SEALDataset'
    val_dataset = eval(dataset_class)(
        path,
        val_data,
        pos_val_edge,
        neg_val_edge,
        num_hops=args.num_hops,
        percent=val_percent,
        split='valid',
        use_coalesce=use_coalesce,
        node_label=args.node_label,
        ratio_per_hop=args.ratio_per_hop,
        max_nodes_per_hop=args.max_nodes_per_hop,
        max_dist=args.max_dist,
        directed=directed,
        sign=args.sign_k > 0,
        k=args.sign_k,
        pi_dim=args.pi_dim,
        filtration=args.filtration,
        extend=args.extend,
        walk_length=args.walk_length
    )
    dataset_class = 'SEALDynamicDataset' if args.dynamic_test else 'SEALDataset'
    test_dataset = eval(dataset_class)(
        path,
        test_data,
        pos_test_edge,
        neg_test_edge,
        num_hops=args.num_hops,
        percent=test_percent,
        split='test',
        use_coalesce=use_coalesce,
        node_label=args.node_label,
        ratio_per_hop=args.ratio_per_hop,
        max_nodes_per_hop=args.max_nodes_per_hop,
        max_dist=args.max_dist,
        directed=directed,
        sign=args.sign_k > 0,
        k=args.sign_k,
        pi_dim=args.pi_dim,
        filtration=args.filtration,
        extend=args.extend,
        walk_length=args.walk_length
    )
    return train_dataset, val_dataset, test_dataset


def get_seal_pos_neg_edges(split, split_edge, edge_index, num_nodes, percent=100):
    if 'edge' in split_edge['train']:
        pos_edge = split_edge[split]['edge'].t()
        if split == 'train':
            new_edge_index, _ = add_self_loops(edge_index)
            neg_edge = negative_sampling(
                new_edge_index, num_nodes=num_nodes,
                num_neg_samples=pos_edge.size(1))
        else:
            neg_edge = split_edge[split]['edge_neg'].t()
        # subsample for pos_edge
        np.random.seed(123)
        num_pos = pos_edge.size(1)
        perm = np.random.permutation(num_pos)
        perm = perm[:int(percent / 100 * num_pos)]
        pos_edge = pos_edge[:, perm]
        # subsample for neg_edge
        np.random.seed(123)
        num_neg = neg_edge.size(1)
        perm = np.random.permutation(num_neg)
        perm = perm[:int(percent / 100 * num_neg)]
        neg_edge = neg_edge[:, perm]

    elif 'source_node' in split_edge['train']:
        source = split_edge[split]['source_node']
        target = split_edge[split]['target_node']
        if split == 'train':
            target_neg = torch.randint(0, num_nodes, [target.size(0), 1],
                                       dtype=torch.long)
        else:
            target_neg = split_edge[split]['target_node_neg']
        # subsample
        np.random.seed(123)
        num_source = source.size(0)
        perm = np.random.permutation(num_source)
        perm = perm[:int(percent / 100 * num_source)]
        source, target, target_neg = source[perm], target[perm], target_neg[perm, :]
        pos_edge = torch.stack([source, target])
        neg_per_target = target_neg.size(1)
        neg_edge = torch.stack([source.repeat_interleave(neg_per_target),
                                target_neg.view(-1)])
    return pos_edge, neg_edge


def k_hop_subgraph(src, dst, num_hops, A, sample_ratio=1.0,
                   max_nodes_per_hop=None, node_features=None,
                   y=1, directed=False, A_csc=None, ricci=None):
    nodes = [src, dst]
    dists = [0, 0]
    visited = set([src, dst])
    fringe = set([src, dst])
    for hop in range(1, num_hops + 1):
        if not directed:
            fringe = neighbors(fringe, A)
        else:
            out_neighbors = neighbors(fringe, A)
            in_neighbors = neighbors(fringe, A_csc, False)
            fringe = out_neighbors.union(in_neighbors)
        fringe = fringe - visited
        visited = visited.union(fringe)
        if sample_ratio < 1.0:
            fringe = random.sample(fringe, int(sample_ratio * len(fringe)))
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(fringe):
                fringe = random.sample(fringe, max_nodes_per_hop)
        if len(fringe) == 0:
            break
        nodes = nodes + list(fringe)
        dists = dists + [hop] * len(fringe)
    # this will permute the rows and columns of the input graph and so the features must also be permuted
    subgraph = A[nodes, :][:, nodes]

    # Remove target link between the subgraph. Works as the first two elements of nodes are the src and dst node
    # this can throw warnings as csr sparse matrices aren't efficient for removing edges, but these graphs are quite sml
    subgraph[0, 1] = 0
    subgraph[1, 0] = 0

    if isinstance(node_features, list):
        node_features = [feat[nodes] for feat in node_features]
    elif node_features is not None:
        node_features = node_features[nodes]

    if ricci is not None:
        ricci = ricci[nodes, :][:, nodes]
    else:
        ricci = None

    return nodes, subgraph, dists, node_features, y, ricci


def construct_pyg_graph(node_ids, adj, dists, node_features, y, ricci, node_label='drnl', max_dist=1000,
                        src_degree=None, dst_degree=None):
    u, v, r = ssp.find(adj)
    num_nodes = adj.shape[0]

    node_ids = torch.LongTensor(node_ids)
    u, v = torch.LongTensor(u), torch.LongTensor(v)
    r = torch.LongTensor(r)
    edge_index = torch.stack([u, v], 0)
    edge_weight = r.to(torch.float)
    y = torch.tensor([y])
    if node_label == 'drnl':  # DRNL
        z = drnl_node_labeling(adj, 0, 1, max_dist)
    elif node_label == 'hop':  # mininum distance to src and dst
        z = torch.tensor(dists)
    elif node_label == 'zo':  # zero-one labeling trick
        z = (torch.tensor(dists) == 0).to(torch.long)
    elif node_label == 'de':  # distance encoding
        z = de_node_labeling(adj, 0, 1, max_dist)
    elif node_label == 'de+':
        z = de_plus_node_labeling(adj, 0, 1, max_dist)
    elif node_label == 'degree':  # this is technically not a valid labeling trick
        z = torch.tensor(adj.sum(axis=0)).squeeze(0)
        z[z > 100] = 100  # limit the maximum label to 100
    else:
        z = torch.zeros(len(dists), dtype=torch.long)
    data = Data(node_features, edge_index, edge_weight=edge_weight, y=y, z=z,
                node_id=node_ids, num_nodes=num_nodes, src_degree=src_degree, dst_degree=dst_degree, ricci=ricci)
    return data


def extract_enclosing_subgraphs(link_index, A, x, y, num_hops, node_label='drnl',
                                ratio_per_hop=1.0, max_nodes_per_hop=None, max_dist=1000,
                                directed=False, A_csc=None, pi_dim=50, filtration_method='degree',
                                extend=False, walk_length=32, ricci=None):
    data_list = []
    transform = T.Compose([
        AddPersistenceImages(pi_dim=pi_dim, filtration_method=filtration_method, extend=extend),
        T.AddRandomWalkPE(walk_length=walk_length, attr_name='pe')
    ])
    for src, dst in tqdm(link_index.tolist()):
        src_degree, dst_degree = get_src_dst_degree(src, dst, A, max_nodes_per_hop)
        tmp = k_hop_subgraph(src, dst, num_hops, A, ratio_per_hop,
                             max_nodes_per_hop, node_features=x, y=y,
                             directed=directed, A_csc=A_csc, ricci=ricci)
        data = construct_pyg_graph(*tmp, node_label, max_dist, src_degree, dst_degree)
        data = transform(data)
        data_list.append(data)

    return data_list


def get_src_dst_degree(src, dst, A, max_nodes):
    src_degree = A[src].sum() if (max_nodes is None or A[src].sum() <= max_nodes) else max_nodes
    dst_degree = A[dst].sum() if (max_nodes is None or A[src].sum() <= max_nodes) else max_nodes
    return src_degree, dst_degree


def neighbors(fringe, A, outgoing=True):
    if outgoing:
        res = set(A[list(fringe)].indices)
    else:
        res = set(A[:, list(fringe)].indices)

    return res


def get_pos_neg_edges(data, sample_frac=1):
    device = data.edge_index.device
    edge_index = data['edge_label_index'].to(device)
    labels = data['edge_label'].to(device)
    pos_edges = edge_index[:, labels == 1].t()
    neg_edges = edge_index[:, labels == 0].t()
    if sample_frac != 1:
        n_pos = pos_edges.shape[0]
        np.random.seed(123)
        perm = np.random.permutation(n_pos)
        perm = perm[:int(sample_frac * n_pos)]
        pos_edges = pos_edges[perm, :]
        neg_edges = neg_edges[perm, :]
    return pos_edges.to(device), neg_edges.to(device)


def compute_ricci_curvature(data):
    Gd = nx.Graph()
    Gd.add_edges_from(data.edge_index.t().tolist())
    Gd_OT = OllivierRicci(Gd, alpha=0.5, method="Sinkhorn", verbose="INFO")
    Gd_OT.compute_ricci_curvature()

    edge_index, edge_weight = [[], []], []
    for n1, n2 in Gd_OT.G.edges():
        edge_index[0].append(n1)
        edge_index[1].append(n2)
        edge_weight.append(Gd_OT.G[n1][n2]['ricciCurvature'])
    ricci = ssp.csr_matrix(
        (edge_weight, (edge_index[0], edge_index[1])),
        shape=(data.num_nodes, data.num_nodes)
    )

    return ricci
