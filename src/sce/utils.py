
from torch_geometric.datasets import Amazon, Planetoid
import torch_geometric.transforms as T
from torch_sparse import SparseTensor, cat
import torch.utils.data
from torch_scatter import scatter_add
from torch_geometric.utils import add_self_loops, remove_self_loops, remove_isolated_nodes
from torch_geometric.utils.num_nodes import maybe_num_nodes
from typing import List, Optional, Tuple, Union
import torch
import numpy as np
from torch_geometric.datasets import Planetoid, CitationFull
import scipy.sparse as sp

from torch_geometric.utils import add_self_loops, degree, to_scipy_sparse_matrix

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


def normalize_adj(edge_index, N):
    idx, val = get_sym(edge_index, self_loops=True, num_nodes=N)
    sparse_A_hat = torch.sparse.FloatTensor(idx, val, torch.Size([N, N]))
    return sparse_A_hat


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


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index



def load_adj_neg(num_nodes, sample):


    row = np.repeat(range(num_nodes), sample)
    col = np.random.randint(0, num_nodes, size=num_nodes * sample)
    new_col = np.concatenate((col, row), axis=0)
    new_row = np.concatenate((row, col), axis=0)
    data = np.ones(new_col.shape[0])
    adj_neg = sp.coo_matrix((data, (new_row, new_col)), shape=(num_nodes, num_nodes))
    adj = np.array(adj_neg.sum(1)).flatten()
    adj_neg = sp.diags(adj) - adj_neg

    return adj_neg.toarray()


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


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

