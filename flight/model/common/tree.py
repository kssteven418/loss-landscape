import numpy as np
import torch
from torch import DoubleTensor
from torch.nn import Parameter
from .linear import SparseLinear, MultiLinear, MultiSumLinear


class Tree(torch.nn.Module):

    def __init__(self, params, opt_level, random_init, dropout, verbose=1):
        super(Tree, self).__init__()
        self.opt_level = opt_level
        self.num_trees = params['w2'].shape[0]
        self.num_features = params['w1'].shape[0]
        self.num_nonterm_nodes = params['w2'].shape[1]
        self.num_leaves = params['w2'].shape[2]

        if verbose >= 1:
            print('num_trees: %d' % self.num_trees)
            print('num_features: %d' % self.num_features)
            print('num_leaves: %d' % self.num_leaves)

        self.predicates = SparseLinear(self.num_features, self.num_nonterm_nodes * self.num_trees)
        if opt_level <= 2 or not random_init:
            self.predicates.weight = Parameter(torch.from_numpy(params['w1']), requires_grad=True if opt_level > 2 else False)
        elif verbose >= 1:
            print("randomly initialize predicates.weight")
        if opt_level <= 1 or not random_init:
            self.predicates.bias = Parameter(torch.from_numpy(params['b1']), requires_grad=True if opt_level > 1 else False)
        elif verbose >= 1:
            print("randomly initialize predicates.bias")

        self.leaf_probs = MultiLinear(self.num_trees, self.num_nonterm_nodes, self.num_leaves)
        if opt_level <= 3 or not random_init:
            self.leaf_probs.weight = Parameter(torch.from_numpy(params['w2']), requires_grad=True if opt_level > 3 else False)
            self.leaf_probs.bias = Parameter(torch.from_numpy(params['b2']), requires_grad=True if opt_level > 3 else False)
        elif verbose >= 1:
            print("randomly initialize leaf_probs.weight")
            print("randomly initialize leaf_probs.bias")

        self.aggregate_leaves = MultiSumLinear(self.num_trees, self.num_leaves, 1, bias=False)
        if opt_level <= 0 or not random_init:
            self.aggregate_leaves.weight = Parameter(torch.from_numpy(params['w3']), requires_grad=True if opt_level > 0 else False)
        elif verbose >= 1:
            print("randomly initialize aggregate_leaves.weight")

        if dropout is not None:
            self.dropout = torch.nn.Dropout(p=dropout)
        else:
            self.dropout = lambda x: x
        self.eps = 1e-36

    def compute_leaf_probs(self, x):
        # sharp sigmoid? e.g., sigma(20 * x)
        if self.opt_level > 1:
            predicates = self.predicates(x).sigmoid().reshape(-1, self.num_trees, self.num_nonterm_nodes).transpose(0, 1)
            predicates = self.dropout(predicates)
            leaf_probs = self.leaf_probs(predicates).sigmoid()

        # step function: (clamp(-eps, eps) + eps) / 2eps
        else:
            predicates = (self.predicates(x).clamp(-self.eps, self.eps) + self.eps) / 2 / self.eps
            predicates = predicates.reshape(-1, self.num_trees, self.num_nonterm_nodes).transpose(0, 1)
            predicates = self.dropout(predicates)
            leaf_probs = (self.leaf_probs(predicates).clamp(-self.eps, self.eps) + self.eps) / 2 / self.eps

        return leaf_probs

    def forward(self, x):
        leaf_probs = self.compute_leaf_probs(x)
        return self.aggregate_leaves(leaf_probs)
