# Neural network implementation

from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d, Flatten, Module
from torch_scatter import scatter_mean

class Net(Module):

    def __init__(self):
        super(Net, self).__init__()
        self.layer = Seq(Lin(30, 32),
                         Lin(32, 200), # Double, Fix the number of neurons
                         BatchNorm1d(200),
                         Lin(200, 200),
                         BatchNorm1d(200),
                         Lin(200, 200),
                         BatchNorm1d(200),
                         Lin(200, 64),
                         BatchNorm1d(64),
                         Lin(64, 32),
                         BatchNorm1d(32),
                         ReLU(),
                         Lin(32,1)
                    )

    def forward(self, x, batch):
        out = scatter_mean(x, batch, dim=0)
        return self.layer(out)