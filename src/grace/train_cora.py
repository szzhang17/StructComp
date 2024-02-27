import torch
import argparse
import numpy as np
from classification import classify
from networks import GCN, MLP, Model
import os
import torch
from torch import Tensor
from torch_scatter import scatter
import random
import numpy as np
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
parser.add_argument('--epochs', type=int, default=20,
                    help='maximum number of epochs')
parser.add_argument('--alpha', type=int, default=5e-4)
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

view1 = torch.zeros([args.cluster, args.num_features])
for j in range(args.cluster):
    view1[j, :] = torch.mean(F_ori[perm[partptr[j]:partptr[j + 1]], :], dim=0)
view1 = view1.to(device)

data = data.to(device)
encoder = MLP(args).to(device)
model = Model(encoder, args.output,  args.output, 0.5).to(device)
model_test = GCN(args).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
model.train()

for epoch in range(1, args.epochs+1):

    view2 = torch.zeros([args.cluster, args.num_features])
    for j in range(args.cluster):
        view2[j, :] = torch.mean(
            F_ori[random.sample(perm[partptr[j]:partptr[j + 1]], (partptr[j + 1] - partptr[j]) // 2), :],
            dim=0)
    view2 = view2.to(device)

    optimizer.zero_grad()
    z1 = model(view1)
    z2 = model(view2)

    loss = model.loss(z1, z2)

    loss.backward()
    optimizer.step()
    # print('------------')

    if epoch%10==0 or epoch==1:

        old_state_dict = model.state_dict()
        new_state_dict = {}

        # 遍历原始 state_dict
        for k, v in old_state_dict.items():
            if k == 'encoder.lin1.bias':
                new_state_dict['conv1.bias'] = v
            elif k == 'encoder.lin2.bias':
                new_state_dict['conv2.bias'] = v
            elif k == 'encoder.lin1.weight':
                new_state_dict['conv1.lin.weight'] = v
            elif k == 'encoder.lin2.weight':
                new_state_dict['conv2.lin.weight'] = v
            elif k == 'encoder.prelu.weight':
                new_state_dict['prelu.weight'] = v
            #else:
                #new_state_dict[k] = v

        # 将新的 state_dict 加载到 model_test 中
        model_test.load_state_dict(new_state_dict)
        with torch.no_grad():
            emb = model_test(data.x, data.edge_index)
        classify(args.dataset, emb.cpu(), data.y.cpu(), 50)


        print('-----------------')

