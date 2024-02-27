import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv





#photo和computer只卷1层

class MLP(torch.nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.lin1 = Linear(args.num_features,  args.output, bias=True)
        #self.lin2 = Linear(args.output, args.output, bias=True)
        self.prelu = torch.nn.PReLU(args.output)

    def forward(self, x):

        z = self.lin1(x)
        #z = self.lin2(z)
        z = self.prelu(z)

        return z



class GCN(torch.nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(args.num_features, args.output, bias=True)
        #self.conv2 = GCNConv(args.output,  args.output, bias=True)
        self.prelu = torch.nn.PReLU(args.output)

    def forward(self, x, edge_index):

        z = self.conv1(x, edge_index)
        #z = self.conv2(z, edge_index)
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


class Model(torch.nn.Module):
    def __init__(self, encoder: MLP, num_hidden: int, num_proj_hidden: int,
                 tau: float = 0.5):
        super(Model, self).__init__()
        self.encoder: MLP = encoder
        self.tau: float = tau

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))


    def loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True, batch_size: int = 0):
        h1 = self.projection(z1)
        h2 = self.projection(z2)


        l1 = self.semi_loss(h1, h2)
        l2 = self.semi_loss(h2, h1)



        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret



class Model2(torch.nn.Module):
    def __init__(self, encoder: GCN, num_hidden: int, num_proj_hidden: int,
                 tau: float = 0.5):
        super(Model2, self).__init__()
        self.encoder: GCN = encoder
        self.tau: float = tau

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)


    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))


    def loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True, batch_size: int = 0):
        h1 = self.projection(z1)
        h2 = self.projection(z2)


        l1 = self.semi_loss(h1, h2)
        l2 = self.semi_loss(h2, h1)



        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret