import torch
from torch.nn.parameter import Parameter

from common.tree import Tree
from common.ohe import OneHotEncoding


class Model(torch.nn.Module):

    def __init__(self, tree_params, opt_level, random_init=False, train_ohe=False,
                 dropout=None, dropout_ohe=None):
        super(Model, self).__init__()

        self.onehot = OneHotEncoding([12, 31, 7, 21, 308, 315], is_train=train_ohe, random_init=random_init, dropout=dropout_ohe)
        self.tree = Tree(tree_params, opt_level, random_init, dropout)

    def forward(self, x_dense, x_sparse):
        xs = self.onehot(x_sparse) # (bs, 12 + 31 + 7 + 21 + 308 + 315)
        xs.insert(0, x_dense)
        return self.tree(torch.cat(xs, dim=1))
