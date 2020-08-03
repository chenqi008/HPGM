import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
from miscc.config import cfg


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, output_dim):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, output_dim)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = self.gc2(x, adj)
        return x


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if cfg.CUDA:
            self.weight = Parameter(torch.cuda.FloatTensor(in_features, out_features))
        else:
            self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            if cfg.CUDA:
                self.bias = Parameter(torch.cuda.FloatTensor(out_features))
            else:
                self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # support = torch.mm(input, self.weight)
        support = torch.matmul(input, self.weight)
        # output = torch.spmm(adj, support)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class BBOX_NET(nn.Module):
    def __init__(self, dim_list, activation='relu', batch_norm='none',
                  dropout=0, final_nonlinearity=True):
        super(BBOX_NET, self).__init__()

        self.mlp = self.build_mlp(dim_list=dim_list, 
            activation=activation, batch_norm=batch_norm,
            dropout=dropout, final_nonlinearity=final_nonlinearity)

    def build_mlp(self, dim_list, activation='relu', batch_norm='none',
                  dropout=0, final_nonlinearity=True):
      layers = []
      for i in range(len(dim_list) - 1):
        dim_in, dim_out = dim_list[i], dim_list[i + 1]
        layers.append(nn.Linear(dim_in, dim_out))
        final_layer = (i == len(dim_list) - 2)
        if not final_layer or final_nonlinearity:
          if batch_norm == 'batch':
            layers.append(nn.BatchNorm1d(dim_out))
          if activation == 'relu':
            layers.append(nn.ReLU())
          elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        if dropout > 0:
          layers.append(nn.Dropout(p=dropout))
      return nn.Sequential(*layers)

    if cfg.TRAIN.USE_GCN:
        def forward(self, objs_vector, graph_objs_vector):
            # element-wise add
            x = torch.add(objs_vector, graph_objs_vector)
            output = self.mlp(x)
            return output
    else:
        def forward(self, objs_vector):
            output = self.mlp(objs_vector)
            return output


def build_mlp(dim_list, activation='relu', batch_norm='none',
              dropout=0, final_nonlinearity=True):
    layers = []
    for i in range(len(dim_list) - 1):
        dim_in, dim_out = dim_list[i], dim_list[i + 1]
        layers.append(nn.Linear(dim_in, dim_out))
        final_layer = (i == len(dim_list) - 2)
        if not final_layer or final_nonlinearity:
          if batch_norm == 'batch':
            layers.append(nn.BatchNorm1d(dim_out))
          if activation == 'relu':
            layers.append(nn.ReLU())
          elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)

