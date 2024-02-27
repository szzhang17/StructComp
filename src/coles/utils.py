

from torch_geometric.datasets import Amazon, Planetoid

import torch.utils.data

from torch_sparse import SparseTensor, cat
import torch.utils.data
from torch_geometric.datasets import Planetoid, CitationFull
import scipy.sparse as sp
from torch_scatter import scatter_add
from torch_geometric.utils import add_self_loops, remove_self_loops, remove_isolated_nodes
from torch_geometric.utils.num_nodes import maybe_num_nodes
from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
from torch_scatter import scatter
import numpy as np
def coalesce(
    edge_index: Tensor,
    edge_attr: Optional[Union[Tensor, List[Tensor]]] = None,
    num_nodes: Optional[int] = None,
    reduce: str = "add",
    is_sorted: bool = False,
    sort_by_row: bool = True,
) -> Union[Tensor, Tuple[Tensor, Tensor], Tuple[Tensor, List[Tensor]]]:

    nnz = edge_index.size(1)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    idx = edge_index.new_empty(nnz + 1)
    idx[0] = -1
    idx[1:] = edge_index[1 - int(sort_by_row)]
    idx[1:].mul_(num_nodes).add_(edge_index[int(sort_by_row)])

    if not is_sorted:
        idx[1:], perm = idx[1:].sort()
        edge_index = edge_index[:, perm]
        if edge_attr is not None and isinstance(edge_attr, Tensor):
            edge_attr = edge_attr[perm]
        elif edge_attr is not None:
            edge_attr = [e[perm] for e in edge_attr]

    mask = idx[1:] > idx[:-1]

    # Only perform expensive merging in case there exists duplicates:
    if mask.all():
        return edge_index if edge_attr is None else (edge_index, edge_attr)

    edge_index = edge_index[:, mask]
    if edge_attr is None:
        return edge_index

    dim_size = edge_index.size(1)
    idx = torch.arange(0, nnz, device=edge_index.device)
    idx.sub_(mask.logical_not_().cumsum(dim=0))

    if isinstance(edge_attr, Tensor):
        edge_attr = scatter(edge_attr, idx, 0, None, dim_size, reduce)
    else:
        edge_attr = [
            scatter(e, idx, 0, None, dim_size, reduce) for e in edge_attr
        ]

    return edge_index, edge_attr


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def normalize_adj(adj):
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)

def partition(edge_index, N, cluster):
    adj = SparseTensor(row=edge_index[0], col=edge_index[1],
        # value=torch.ones(E, device=data.edge_index.device),
        sparse_sizes=(N, N))
    recursive = False
    _, partptr_, perm_ = adj.partition(cluster, recursive)
    partptr = partptr_.tolist()
    perm = perm_.tolist()

    #修正分割结果，部分点会被单独且重复分为1类
    partptr = list(set(partptr))
    partptr.sort()

    return partptr, perm, len(partptr)-1
def load_adj_neg(num_nodes, sample):

    col = np.random.randint(0, num_nodes, size=num_nodes * sample)
    row = np.repeat(range(num_nodes), sample)
    index = np.not_equal(col,row)
    col = col[index]
    row = row[index]
    new_col = np.concatenate((col,row),axis=0)
    new_row = np.concatenate((row,col),axis=0)
    data = np.ones(new_col.shape[0])
    adj_neg = sp.coo_matrix((data, (new_row, new_col)), shape=(num_nodes, num_nodes))
    adj_neg = normalize_adj(adj_neg)

    return adj_neg.toarray()

def load_dataset(dataset_str):
    if dataset_str == 'cora' or dataset_str == 'pubmed' or dataset_str == 'citeseer':
        dataset = Planetoid(root='./dataset', name=dataset_str)
        data = dataset[0]
    elif dataset_str == 'Amazon-Computers':
        dataset = Amazon(root='./dataset', name='computers')
        data = dataset[0]
    elif dataset_str == 'Amazon-Photo':
        dataset = Amazon(root='./dataset', name='photo')
        data = dataset[0]

    return data

def get_sym(edge_index, self_loops = True, num_nodes: Optional[int] = None,
            edge_weight: Optional[torch.Tensor] = None):

    if self_loops==True:
        edge_index = add_self_loops(edge_index)[0]

    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)

    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return edge_index, edge_weight


def normalize_adj_sp(edge_index, N, loops):
    idx, val = get_sym(edge_index, self_loops=loops, num_nodes=N)
    sparse_A_hat = torch.sparse.FloatTensor(idx, val, torch.Size([N, N]))
    return sparse_A_hat



def load_adj_lap(edge_index, N):
    adj_normalized = normalize_adj_sp(edge_index, N, loops=True).to_dense()

    Laplacian = normalize_adj_sp(edge_index, N, loops=False).to_dense()

    #adj_normalized = torch.from_numpy(normalize_adj(sp.eye(adj.shape[0]) + adj).toarray()).float()
    #Laplacian = torch.from_numpy(sp.eye(adj.shape[0]) - normalize_adj(adj).toarray()).float()
    #Laplacian = torch.from_numpy(normalize_adj(adj).toarray()).float()

    return adj_normalized, Laplacian
