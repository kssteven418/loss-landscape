import torch
from torch.nn import Parameter
from torch.nn import init


class OneHotEncoding(torch.nn.Module):

    def __init__(self, sizes, max_size=None, is_train=False, random_init=False, dropout=None, verbose=1):
        super(OneHotEncoding, self).__init__()
        embeddings = []
        if random_init and verbose >= 1:
            print("Randomly initialize Embedding for OneHotEncoding")

        for size in sizes:
            weight = torch.zeros([size + 1, size if max_size is None else min(max_size, size)], dtype=torch.float64)
            for i in range(size):
                if max_size is not None and i >= max_size:
                    weight[i, i % max_size] = 1
                else:
                    weight[i, i] = 1
            embedding = torch.nn.Embedding(weight.shape[0], weight.shape[1], sparse=True)
            if not random_init:
                embedding.weight = Parameter(weight, requires_grad=is_train)
            else:
                assert is_train
                embedding = embedding.double()
            embeddings.append(embedding)
        self.embeddings = torch.nn.ModuleList(embeddings)

        if dropout is not None:
            self.dropout = torch.nn.Dropout(p=dropout)
        else:
            self.dropout = lambda x: x

    def forward(self, x):
        assert len(self.embeddings) == x.shape[1]
        result = []
        for i, embedding in enumerate(self.embeddings):
            result.append(self.dropout(embedding(x[:, i])))
        return result
