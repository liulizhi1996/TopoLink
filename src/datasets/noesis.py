from typing import Optional, Callable, List

import os.path as osp
import scipy.io

import torch
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.utils import from_scipy_sparse_matrix


class NOESISDataset(InMemoryDataset):
    """
    Original datasets are collected by NOESIS.
    https://noesis.ikor.org/datasets/link-prediction
    """
    url = 'https://github.com/Barcavin/efficient-node-labelling/raw/master/data'

    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 force_reload: bool = False):
        self.name = name

        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        return [f'{self.name}.mat']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        for name in self.raw_file_names:
            download_url('{}/{}'.format(self.url, name), self.raw_dir)

    def process(self):
        data = read_data(self.raw_dir, self.name)
        data = data if self.pre_transform is None else self.pre_transform(data)
        self.save([data], self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}()'


def read_data(folder, prefix):
    path = osp.join(folder, '{}.mat'.format(prefix))
    mat = scipy.io.loadmat(path)
    net = mat['net']
    edge_index = from_scipy_sparse_matrix(net)[0]
    num_nodes = torch.max(edge_index).item() + 1
    x = torch.ones((num_nodes, 1), dtype=torch.float)
    y = torch.ones((num_nodes,), dtype=torch.int)
    data = Data(x=x, y=y, edge_index=edge_index, num_nodes=num_nodes)
    return data
