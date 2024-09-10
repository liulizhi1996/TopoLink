from typing import Optional

import torch
from torch import Tensor
from torch.nn import Linear

from torch_geometric.nn.aggr import Aggregation


class PairAggregation(Aggregation):
    def __init__(self, channels):
        super().__init__()
        self.linear = Linear(channels, channels)

    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        x, _ = self.to_dense_batch(x, index, ptr, dim_size, dim)
        x_src, x_dst = x[:, 0, :].squeeze(), x[:, 1, :].squeeze()
        out = self.linear(x_src * x_dst)
        return torch.relu(out)
