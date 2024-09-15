import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearHead(nn.Module):
    def __init__(self, net, dim_in=2048, dim_out=1000):
        super().__init__()
        self.net = net
        self.fc = nn.Linear(dim_in, dim_out)

        for param in self.net.parameters():
            param.requires_grad = False

        self.fc.weight.data.normal_(mean=0.0, std=0.01)
        self.fc.bias.data.zero_()

    def forward(self, x):
        with torch.no_grad():
            feat = self.net(x)
        return self.fc(feat)


class ProjectionHead(nn.Module):
    def __init__(self, dim_in=2048, dim_out=128, dim_hidden=2048):
        super().__init__()
        self.linear1 = nn.Linear(dim_in, dim_hidden)
        self.bn1 = nn.BatchNorm1d(dim_hidden)
        self.relu1 = nn.ReLU(True)
        self.linear2 = nn.Linear(dim_hidden, dim_hidden)
        self.bn2 = nn.BatchNorm1d(dim_hidden)
        self.relu2 = nn.ReLU(True)
        self.linear3 = nn.Linear(dim_hidden, dim_out)
        
    def forward(self, x):
        x = self.linear1(x) #.unsqueeze(-1).unsqueeze(-1)
        x = self.bn1(x).squeeze(-1).squeeze(-1)
        x = self.relu1(x)
        x = self.linear2(x) #.unsqueeze(-1).unsqueeze(-1)
        x = self.bn2(x).squeeze(-1).squeeze(-1)
        x = self.relu2(x)
        x = self.linear3(x)
        return x

class ProjectionHead2d(nn.Module):
    def __init__(self, in_dim, use_bn=True, nlayers=3, hidden_dim=4096, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Conv2d(in_dim, bottleneck_dim, 1)
        else:
            layers = [nn.Conv2d(in_dim, hidden_dim, 1)]
            if use_bn:
                layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Conv2d(hidden_dim, hidden_dim, 1))
                if use_bn:
                    layers.append(nn.BatchNorm2d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Conv2d(hidden_dim, bottleneck_dim, 1))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        return x

