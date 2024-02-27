import torch
from utils import load_adj_neg, load_adj_lap, load_dataset, partition, coalesce
from ssgc import Net
import argparse
import numpy as np
from classification import classify

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='cora',
                    help='dataset')
parser.add_argument('--seed', type=int, default=123,
                    help='seed')
parser.add_argument('--nhid', type=int, default=512,
                    help='hidden size')
parser.add_argument('--output', type=int, default=512,
                    help='output size')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='weight decay')
parser.add_argument('--epochs', type=int, default=20,
                    help='maximum number of epochs')
parser.add_argument('--sample', type=int, default=5,
                    help='    ')
parser.add_argument('--num_nodes', type=int, default=2708,
                    help='    ')
parser.add_argument('--num_features', type=int, default=1433,
                    help='    ')
parser.add_argument('--cluster', type=int, default=300,
                    help='    ')
args = parser.parse_args()
args.device = 'cpu'
torch.manual_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = load_dataset(args.dataset)
N, E, args.num_features = data.num_nodes, data.num_edges,  data.num_features
args.num_nodes = N
adj_normalized, _ = load_adj_lap(data.edge_index, N)
#print(adj_normalized)
adj_normalized = adj_normalized.to(device)
F = data.x.to(device)
for i in range(2):
    F = torch.mm(adj_normalized, F)

partptr, perm, args.cluster = partition(data.edge_index, N, cluster=args.cluster)
F_ori = data.x.to(device)
s = torch.zeros([args.cluster, args.num_features])
for j in range(args.cluster):
    s[j, :] = torch.mean(F_ori[perm[partptr[j]:partptr[j + 1]], :], dim=0)
s = s.to(device)


sx = torch.zeros(N)
for j in range(args.cluster):
    sx[perm[partptr[j]:partptr[j + 1]]]=j
coarsening_edge = torch.cat((sx[data.edge_index[0]].view(1, -1), sx[data.edge_index[1]].view(1, -1)), dim=0).long()
edge_weight = torch.ones(data.edge_index.size(1))
coarsening_edge, _ = coalesce(coarsening_edge, edge_weight, reduce="sum")
_, c_lap_normalized = load_adj_lap(coarsening_edge, args.cluster)
c_lap_normalized = c_lap_normalized.to(device)

neg_sample = torch.from_numpy(load_adj_neg(args.cluster, args.sample)).float().to(device)

model = Net(args).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
model.train()
Lambda = 0.8

for epoch in range(1, args.epochs+1):

    optimizer.zero_grad()
    out = model(s)

    loss = (Lambda*torch.trace(torch.mm(torch.mm(torch.transpose(out, 0, 1), neg_sample), out)) - torch.trace(
        torch.mm(torch.mm(torch.transpose(out, 0, 1), c_lap_normalized), out)))/out.shape[0]

    #print(loss)
    loss.backward()
    optimizer.step()

    if epoch%10==0:
        with torch.no_grad():
            emb = model(F)
        classify(args.dataset, emb.cpu(), data.y.cpu(), 50)

