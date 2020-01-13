import math
import torch
from torch.nn import init
from torch.nn.parameter import Parameter


class LinearBase(torch.nn.Module):

    def __init__(self, in_features, out_features):
        super(LinearBase, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)


class SparseLinear(LinearBase):

    def __init__(self, in_features, out_features, bias=True):
        super(SparseLinear, self).__init__(in_features, out_features)
        self.weight = Parameter(torch.DoubleTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.DoubleTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, input):
        output = input.matmul(self.weight)
        if self.bias is not None:
            output += self.bias
        return output


class MultiLinear(LinearBase):

    def __init__(self, num_layers, in_features, out_features, bias=True):
        super(MultiLinear, self).__init__(in_features, out_features)
        self.num_layers = num_layers
        self.weight = Parameter(torch.DoubleTensor(num_layers, in_features, out_features))
        if bias:
            self.bias = Parameter(torch.DoubleTensor(num_layers, 1, out_features))
        else:
            self.register_parameter('bias', None)
            self.zero_bias = Parameter(torch.zeros(1, dtype=torch.double))
        self.reset_parameters()

    def forward(self, input):
        if self.bias is None:
            return torch.baddbmm(0, self.zero_bias, 1, input, self.weight)
        else:
            return torch.baddbmm(self.bias, input, self.weight)


class MultiSumLinear(LinearBase):

    def __init__(self, num_layers, in_features, out_features, bias=True):
        super(MultiSumLinear, self).__init__(in_features, out_features)
        self.num_layers = num_layers
        self.weight = Parameter(torch.DoubleTensor(num_layers, in_features, out_features))
        if bias:
            self.bias = Parameter(torch.DoubleTensor(1, out_features))
        else:
            self.register_parameter('bias', None)
            self.zero_bias = Parameter(torch.zeros(1, dtype=torch.double))
        self.reset_parameters()

    def forward(self, input):
        if self.bias is None:
            return torch.addbmm(0, self.zero_bias, 1, input, self.weight)
        else:
            return torch.addbmm(self.bias, input, self.weight)
