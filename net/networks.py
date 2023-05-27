import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CausalMask(nn.Module):
    def __init__(self, patch_num, channel):
        super(CausalMask, self).__init__()

        self.P = patch_num
        self.channel = channel

        e = torch.cat([torch.zeros(self.P, self.P, 1), torch.ones(self.P, self.P, 1)], dim=-1)
        self.M1_edge = nn.Parameter(Variable(e, requires_grad=True))
        self.M2_edge = nn.Parameter(Variable(e, requires_grad=True))
        n = torch.cat([torch.zeros(self.P, 1), torch.ones(self.P, 1)], dim=-1)
        self.M2_node = nn.Parameter(Variable(n, requires_grad=True))
        self.M1_node = nn.Parameter(Variable(n, requires_grad=True))

    @staticmethod
    def gumbel_softmax(logits, tau: float = 1, hard: bool = True, dim: int = -1):
        gumbel_dist = torch.distributions.gumbel.Gumbel(
            torch.tensor(0., device=logits.device, dtype=logits.dtype),
            torch.tensor(1., device=logits.device, dtype=logits.dtype))
        gumbels = gumbel_dist.sample(logits.shape)

        gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
        y_soft = gumbels.softmax(dim)

        if hard:
            # Straight through.
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            # Reparametrization trick.
            ret = y_soft

        return ret[..., 1]

    @staticmethod
    def hardmax(M, dim: int = -1):
        return M.argmax(dim).float()

    def forward(self, train=True):
        pass
        return masks, ss


class CausalNet(nn.Module):
    def __init__(self, num_class, hidden1, hidden2, kernels=None):
        super(CausalNet, self).__init__()

        self.hidden1 = hidden1
        if kernels is None or kernels == 'None':
            kernels = [3] * len(self.hidden1)
        padding = (lambda x: int((x-1)/2))
        self.patch_emb = nn.Sequential()
        for i in range(len(self.hidden1)):
            if i == 0:
                self.patch_emb.add_module('conv_{}'.format(i), nn.Conv3d(1, self.hidden1[i], kernel_size=kernels[i], padding=padding(kernels[i])))
            else:
                self.patch_emb.add_module('conv_{}'.format(i), nn.Conv3d(self.hidden1[i-1], self.hidden1[i], kernel_size=kernels[i], padding=padding(kernels[i])))
            self.patch_emb.add_module('norm_{}'.format(i), nn.BatchNorm3d(self.hidden1[i]))
            self.patch_emb.add_module('relu_{}'.format(i), nn.ReLU(True))
        self.patch_emb.add_module('avgpool', nn.AdaptiveAvgPool3d(output_size=(1, 1, 1)))

        self.gcns2_causal = ConvGCN(in_features=hidden1[-1], hidden=hidden2)
        self.gcns2_relate = ConvGCN(in_features=hidden1[-1], hidden=hidden2)

        self.mlp_causal = nn.Linear(sum(hidden2) * 2, num_class)
        self.mlp_combine = nn.Linear(sum(hidden2) * 4, num_class)

    def emb(self, x):  # [8, 200, 5, 5, 5]
        x_np = x.reshape(-1, *x.shape[2:]).unsqueeze(1)  # [1600, 1, 5, 5, 5]
        x_conv3d = self.patch_emb(x_np)  # [1600, d, 1, 1, 1]
        x_emb = x_conv3d.reshape(*x.shape[:2], -1)  # [8, 200, d]
        return x_emb  # [B,P,d]

    def prediction_whole(self, x_new, edge):
        xs = self.gcns2_causal(x_new, edge)  # [B,P,d]
        graph = torch.cat([readout(xx) for xx in xs], dim=-1)  # [B,P,d*xx]
        yc = self.mlp_causal(graph)
        return yc

    def prediction_causal(self, x_new, edge, masks):
        pass
        return yc

    def prediction_counterfactual(self, x_new, edge, masks):  # remove causal
        pass
        return yo

    def prediction_combine(self, x_new, edge, masks):  # causal + non-causal
        pass
        yr = torch.tensor([]).to(x_new.device)
        for i in range(N):
            graph = torch.cat([graphC[i, :].repeat(N, 1), graphR], dim=-1)  # [B,P,d*xx]
            yr = torch.cat([yr, self.mlp_combine(graph).unsqueeze(0)])
        return yr


class ConvGCN(nn.Module):
    def __init__(self, in_features, hidden):
        super(ConvGCN, self).__init__()

        self.hidden = hidden
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for i in range(len(self.hidden)):
            if i == 0:
                conv = GCN(in_features=in_features, out_features=self.hidden[0])
            else:
                conv = GCN(in_features=self.hidden[i - 1], out_features=self.hidden[i])
            self.convs.append(conv)
            self.batch_norms.append(nn.BatchNorm1d(self.hidden[i]))

    def forward(self, x, edge):
        xs = []
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge)
            x = F.relu(bn(x.transpose(1, 2)).transpose(1, 2))
            xs.append(x)
        return xs


def readout(x):
    return torch.concat([global_max_pool(x), global_mean_pool(x)], dim=-1)


def global_max_pool(x):  # x:[B,P,d]
    return F.max_pool1d(x.transpose(1, 2), x.shape[1]).squeeze(2)


def global_mean_pool(x):  # x:[B,P,d]
    return F.avg_pool1d(x.transpose(1, 2), x.shape[1]).squeeze(2)


class GCN(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Linear(in_features, out_features, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        self.W.reset_parameters()

    def forward(self, X, adj):  # X:[B*P,d]=[1600,4], adj:[B*P,B*P]=[1600,1600]
        XW = self.W(X)
        AXW = torch.matmul(adj, XW)  # AXW
        return AXW

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_features, self.out_features)
