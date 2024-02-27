import torch
import argparse
import numpy as np
from classification import classify
from networks import Lin

import torch

import random

from utils import load_adj_neg, load_dataset, normalize_adj, partition


parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='cora',
                    help='dataset')
parser.add_argument('--seed', type=int, default=123,
                    help='seed')
parser.add_argument('--output', type=int, default=512,
                    help='output size')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='weight decay')
parser.add_argument('--epochs', type=int, default=50,
                    help='maximum number of epochs')
parser.add_argument('--sample', type=int, default=5,
                    help='    ')
parser.add_argument('--alpha', type=int, default=20000,
                    help='    ')
parser.add_argument('--beta', type=int, default=0,
                    help='    ')
parser.add_argument('--gamma', type=int, default=0,
                    help='    ')
parser.add_argument('--cluster', type=int, default=300,
                    help='    ')
args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = load_dataset(args.dataset)
N, E, args.num_features = data.num_nodes, data.num_edges,  data.num_features
args.num_nodes = N
adj = normalize_adj(data.edge_index, N)

partptr, perm, args.cluster = partition(data.edge_index, N, cluster=args.cluster)


F_ori = data.x.to(device)
F = data.x
for i in range(2):
    F = torch.spmm(adj, F)
F = F.to(device)


s = torch.zeros([args.cluster, args.num_features])
for j in range(args.cluster):
    s[j, :] = torch.mean(F_ori[perm[partptr[j]:partptr[j + 1]], :], dim=0)
s = s.to(device)


model = Lin(args).to(device)
#model = Lin_2_layer(args).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
model.train()

for epoch in range(1, args.epochs+1):

    neg_sample = torch.from_numpy(load_adj_neg(args.cluster, args.sample)).float().to(device)
    optimizer.zero_grad()
    out = model(s)
    loss = args.alpha / torch.trace(torch.mm(torch.mm(torch.transpose(out, 0, 1), neg_sample), out))
    loss.backward()
    optimizer.step()

   # print('------------')
    if epoch%50==0 or epoch==1:
        with torch.no_grad():
            emb = model(F)
        classify(args.dataset, emb.cpu(), data.y.cpu(), 50)
        print('-----------------')

