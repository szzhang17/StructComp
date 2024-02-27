import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv





#photo和computer只卷1层

class MLP(torch.nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.lin1 = Linear(args.num_features,  args.output, bias=True)
        self.lin2 = Linear(args.output, args.output, bias=True)
        self.prelu = torch.nn.PReLU(args.output)

    def forward(self, x):

        z = self.lin1(x)
        z = self.lin2(z)
        z = self.prelu(z)

        return z



class GCN(torch.nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(args.num_features, args.output, bias=True)
        self.conv2 = GCNConv(args.output,  args.output, bias=True)
        self.prelu = torch.nn.PReLU(args.output)

    def forward(self, x, edge_index):

        z = self.conv1(x, edge_index)
        z = self.conv2(z, edge_index)
        z = self.prelu(z)
        return z


class SGC(torch.nn.Module):
    def __init__(self, args):
        super(SGC, self).__init__()
        self.lin = Linear(args.num_features, args.output, bias=True)
        #self.prelu = torch.nn.PReLU(args.output)

    def forward(self, F):

        z = self.lin(F)
        #z = self.prelu(z)

        return z


class Lin(torch.nn.Module):
    def __init__(self, args):
        super(Lin, self).__init__()
        self.lin = Linear(args.num_features, args.output, bias=True)
        #self.prelu = torch.nn.PReLU(args.output)

    def forward(self, F):

        z = self.lin(F)
        #z = self.prelu(z)

        return z