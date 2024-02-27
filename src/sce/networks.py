import torch
import torch.nn.functional as F
from torch.nn import Linear


class Lin(torch.nn.Module):
    def __init__(self, args):
        super(Lin, self).__init__()
        self.lin = Linear(args.num_features, args.output, bias=True)
        #self.prelu = torch.nn.PReLU(args.output)

    def forward(self, F):

        z = self.lin(F)
        #z = self.prelu(z)

        return z


class Lin_2_layer(torch.nn.Module):
    def __init__(self, args):
        super(Lin_2_layer, self).__init__()
        self.lin1 = Linear(args.num_features, args.output, bias=True)
        self.lin2 = Linear(args.output, args.output, bias=True)
    def forward(self, F):

        z = self.lin2(self.lin1(F))

        return z

