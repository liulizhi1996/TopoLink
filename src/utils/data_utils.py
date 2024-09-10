import os

import torch
from ogb.linkproppred import PygLinkPropPredDataset
from torch_geometric.data import Data
from torch_geometric import datasets
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import (add_self_loops, negative_sampling,
                                   to_undirected)
from torch_geometric.loader import DataLoader as pygDataLoader

from src.utils.lcc import get_largest_connected_component, remap_edges, get_node_mapper
from src.utils.dataset_utils import get_train_val_test_datasets
from src.datasets.noesis import NOESISDataset


# root of the project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))


def get_loaders(args, dataset, splits):
    train_data, val_data, test_data = splits['train'], splits['valid'], splits['test']
    train_dataset, val_dataset, test_dataset = get_train_val_test_datasets(dataset, train_data, val_data, test_data,
                                                                           args)

    train_loader = pygDataLoader(train_dataset, batch_size=args.batch_size,
                                 shuffle=True, num_workers=args.num_workers)
    # as the val and test edges are often sampled they also need to be shuffled
    # the citation2 dataset has specific negatives for each positive and so can't be shuffled
    shuffle_val = False if args.dataset_name.startswith('ogbl-citation') else True
    val_loader = pygDataLoader(val_dataset, batch_size=args.batch_size, shuffle=shuffle_val,
                               num_workers=args.num_workers)
    shuffle_test = False if args.dataset_name.startswith('ogbl-citation') else True
    test_loader = pygDataLoader(test_dataset, batch_size=args.batch_size, shuffle=shuffle_test,
                                num_workers=args.num_workers)

    return train_loader, val_loader, test_loader


def get_data(args):
    include_negatives = True
    dataset_name = args.dataset_name
    val_pct = args.val_pct
    test_pct = args.test_pct
    use_lcc_flag = True
    directed = False
    eval_metric = args.eval_metric
    path = os.path.join(ROOT_DIR, 'data')
    print(f'reading data from: {path}')
    if dataset_name.startswith('ogbl'):
        use_lcc_flag = False
        dataset = PygLinkPropPredDataset(name=dataset_name, root=path)
        if dataset_name == 'ogbl-ddi':
            dataset.data.x = torch.ones((dataset.data.num_nodes, 1))
            dataset.data.edge_weight = torch.ones(dataset.data.edge_index.size(1), dtype=int)
    else:
        pyg_dataset_dict = {
            'Cora': (datasets.Planetoid, {'name': 'Cora'}),
            'Citeseer': (datasets.Planetoid, {'name': 'Citeseer'}),
            'Pubmed': (datasets.Planetoid, {'name': 'Pubmed'}),
            'CS': (datasets.Coauthor, {'name': 'CS'}),
            'Physics': (datasets.Coauthor, {'name': 'Physics'}),
            'Computers': (datasets.Amazon, {'name': 'Computers'}),
            'Photo': (datasets.Amazon, {'name': 'Photo'}),
            'Wiki': (datasets.AttributedGraphDataset, {'name': 'Wiki'}),
            'BlogCatalog': (datasets.AttributedGraphDataset, {'name': 'BlogCatalog'}),
            'Facebook': (datasets.AttributedGraphDataset, {'name': 'Facebook'})
        }
        if dataset_name in pyg_dataset_dict:
            dataset_class, kwargs = pyg_dataset_dict[dataset_name]
            dataset = dataset_class(root=path, **kwargs)
        else:
            dataset = NOESISDataset(path, dataset_name)

    # set the metric
    if dataset_name.startswith('ogbl-citation'):
        eval_metric = 'mrr'
        directed = True
    elif dataset_name.startswith('ogbl-vessel'):
        args.eval_metric = 'rocauc'
        directed = False

    if use_lcc_flag:
        dataset = use_lcc(dataset)

    undirected = not directed

    if dataset_name.startswith('ogbl'):  # use the built in splits
        data = dataset[0]
        split_edge = dataset.get_edge_split()
        if dataset_name == 'ogbl-vessel':
            # normalize node features
            data.x[:, 0] = torch.nn.functional.normalize(data.x[:, 0], dim=0)
            data.x[:, 1] = torch.nn.functional.normalize(data.x[:, 1], dim=0)
            data.x[:, 2] = torch.nn.functional.normalize(data.x[:, 2], dim=0)
        if dataset_name == 'ogbl-collab' and args.year > 0:  # filter out training edges before args.year
            data, split_edge = filter_by_year(data, split_edge, args.year)
        splits = get_ogb_data(data, split_edge, dataset_name, args.num_negs)
    else:  # make random splits
        transform = RandomLinkSplit(is_undirected=undirected, num_val=val_pct, num_test=test_pct,
                                    add_negative_train_samples=include_negatives)
        train_data, val_data, test_data = transform(dataset.data)
        splits = {'train': train_data, 'valid': val_data, 'test': test_data}

    return dataset, splits, directed, eval_metric


def filter_by_year(data, split_edge, year):
    selected_year_index = torch.reshape(
        (split_edge['train']['year'] >= year).nonzero(as_tuple=False), (-1,))
    split_edge['train']['edge'] = split_edge['train']['edge'][selected_year_index]
    split_edge['train']['weight'] = split_edge['train']['weight'][selected_year_index]
    split_edge['train']['year'] = split_edge['train']['year'][selected_year_index]
    train_edge_index = split_edge['train']['edge'].t()
    # create adjacency matrix
    new_edges = to_undirected(train_edge_index, split_edge['train']['weight'], reduce='add')
    new_edge_index, new_edge_weight = new_edges[0], new_edges[1]
    data.edge_index = new_edge_index
    data.edge_weight = new_edge_weight.unsqueeze(-1)
    return data, split_edge


def get_ogb_data(data, split_edge, dataset_name, num_negs=1):
    dataset_name = dataset_name.replace('-', '_')
    if num_negs == 1:
        negs_name = f'{ROOT_DIR}/data/{dataset_name}/negative_samples.pt'
    else:
        negs_name = f'{ROOT_DIR}/data/{dataset_name}/negative_samples_{num_negs}.pt'
    print(f'looking for negative edges at {negs_name}')
    if os.path.exists(negs_name):
        print('loading negatives from disk')
        train_negs = torch.load(negs_name)
    else:
        print('negatives not found on disk. Generating negatives')
        train_negs = get_ogb_train_negs(split_edge, data.edge_index, data.num_nodes, num_negs, dataset_name)
        torch.save(train_negs, negs_name)
    splits = {}
    for key in split_edge.keys():
        # the ogb datasets come with test and valid negatives, but you have to cook your own train negs
        neg_edges = train_negs if key == 'train' else None
        edge_label, edge_label_index = make_obg_supervision_edges(split_edge, key, neg_edges)
        # use the validation edges for message passing at test time
        # according to the rules https://ogb.stanford.edu/docs/leader_rules/ only collab can use val edges at test time
        if key == 'test' and dataset_name == 'ogbl_collab':
            vei, vw = to_undirected(split_edge['valid']['edge'].t(), split_edge['valid']['weight'])
            edge_index = torch.cat([data.edge_index, vei], dim=1)
            edge_weight = torch.cat([data.edge_weight, vw.unsqueeze(-1)], dim=0)
        else:
            edge_index = data.edge_index
            if hasattr(data, "edge_weight"):
                edge_weight = data.edge_weight
            else:
                edge_weight = torch.ones(data.edge_index.shape[1])
        splits[key] = Data(x=data.x, edge_index=edge_index, edge_weight=edge_weight, edge_label=edge_label,
                           edge_label_index=edge_label_index)
    return splits


def get_ogb_pos_edges(split_edge, split):
    if 'edge' in split_edge[split]:
        pos_edge = split_edge[split]['edge']
    elif 'source_node' in split_edge[split]:
        pos_edge = torch.stack([split_edge[split]['source_node'], split_edge[split]['target_node']],
                               dim=1)
    else:
        raise NotImplementedError
    return pos_edge


def get_ogb_train_negs(split_edge, edge_index, num_nodes, num_negs=1, dataset_name=None):
    if 'edge_neg' in split_edge['train']:
        # use pre-sampled negative training edges for ogbl-vessel
        neg_edge = split_edge['train']['edge_neg']
        return neg_edge

    pos_edge = get_ogb_pos_edges(split_edge, 'train').t()
    if dataset_name is not None and dataset_name.startswith('ogbl_citation'):
        neg_edge = get_same_source_negs(num_nodes, num_negs, pos_edge)
    else:  # any source is fine
        new_edge_index, _ = add_self_loops(edge_index)
        neg_edge = negative_sampling(
            new_edge_index, num_nodes=num_nodes,
            num_neg_samples=pos_edge.size(1) * num_negs)
    return neg_edge.t()


def get_same_source_negs(num_nodes, num_negs_per_pos, pos_edge):
    print(f'generating {num_negs_per_pos} single source negatives for each positive source node')
    dst_neg = torch.randint(0, num_nodes, (1, pos_edge.size(1) * num_negs_per_pos), dtype=torch.long)
    src_neg = pos_edge[0].repeat_interleave(num_negs_per_pos)
    return torch.cat([src_neg.unsqueeze(0), dst_neg], dim=0)


def make_obg_supervision_edges(split_edge, split, neg_edges=None):
    if neg_edges is not None:
        neg_edges = neg_edges
    else:
        if 'edge_neg' in split_edge[split]:
            neg_edges = split_edge[split]['edge_neg']
        elif 'target_node_neg' in split_edge[split]:
            n_neg_nodes = split_edge[split]['target_node_neg'].shape[1]
            neg_edges = torch.stack([split_edge[split]['source_node'].unsqueeze(1).repeat(1, n_neg_nodes).ravel(),
                                     split_edge[split]['target_node_neg'].ravel()
                                     ]).t()
        else:
            raise NotImplementedError

    pos_edges = get_ogb_pos_edges(split_edge, split)
    n_pos, n_neg = pos_edges.shape[0], neg_edges.shape[0]
    edge_label = torch.cat([torch.ones(n_pos), torch.zeros(n_neg)], dim=0)
    edge_label_index = torch.cat([pos_edges, neg_edges], dim=0).t()
    return edge_label, edge_label_index


def use_lcc(dataset):
    lcc = get_largest_connected_component(dataset)

    x_new = dataset.data.x[lcc]
    y_new = dataset.data.y[lcc]

    row, col = dataset.data.edge_index.numpy()
    edges = [[i, j] for i, j in zip(row, col) if i in lcc and j in lcc]
    edges = remap_edges(edges, get_node_mapper(lcc))

    data = Data(
        x=x_new,
        edge_index=torch.LongTensor(edges),
        y=y_new,
        train_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
        test_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
        val_mask=torch.zeros(y_new.size()[0], dtype=torch.bool)
    )
    dataset.data = data
    return dataset
