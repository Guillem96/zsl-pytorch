from typing import Union, Sequence

import torch
import torch.nn as nn


class LinearSemanticUnit(nn.Module):
    """
    Refer to section 3.2 of https://arxiv.org/pdf/1611.05088.pdf

    Semantic unit that takes a L-dimensional semantic 
    representation vector of the corresponding class y input, and after going 
    through two fully connected + ReLU layers outputs a 
    D-dimensional semantic embedding vector.

    """
    def __init__(self, in_features: int, out_features: int):
        super(LinearSemanticUnit, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.FloatTensor:
        return self.relu(self.linear(x))


class MultiModalSemanticUnit(nn.Module):

    """
    Refer to section 3.3 of https://arxiv.org/pdf/1611.05088.pdf

    Model that maps different semantic representation vectors to a multi-modal 
    fusion layer/space where theyare added. In other words, it applies a linear
    combination to each representation and finally adds them.

    Parameters
    ----------
    n_branches: int
        Number of representations per image.
    in_features: Union[int, Sequence[int]]
        Input features for each branch. If a integer is used, the specified
        value is going to be used for all branches
    out_features: int
        Output features for multimodal aggregation. As the aggregation 
        is an element wise sum and activation layer, out_features must be 
        consistent through all branches
    """
    def __init__(self, 
                 n_branches: int = 2, 
                 features: Union[int, Sequence[int]]):
        super(MultiModalSemanticUnit, self).__init__()

        if isinstance(in_features, int):
            in_features = [in_features] * n_branches
        
        self.n_branches = n_branches        
        self.branches = nn.ModuleList([
            nn.Linear(f, out_features) for f in in_features])
        self.tanh = nn.Tanh()

    def forward(self, x: Sequence[torch.FloatTensor]) -> torch.FloatTensor:
        x = [self.branches[i](x[i]) for i in range(self.n_branches)]
        # http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
        return 1.7159 * self.tanh(2/3 * sum(x))



class BidirectionalSemanticUnit(nn.Module):
    """
    Refer to section 3.4 from https://arxiv.org/pdf/1611.05088.pdf

    When text description is available foreach training image. We extract the 
    semantic representation of the description using an LSTM

    Parameters
    ----------
    embedding_matrix: torch.FloatTensor
        Pretrained embedding matrix
    padding_idx: int
        Padding index in your vocabulary
    out_features: int
        Number of features should the model output
    hidden_size: int
        LSTM parameter
    """
    def __init__(self, 
                 embedding_matrix: torch.FloatTensor, 
                 padding_idx: int,
                 hidden_size: int,
                 out_features: int):

        super(BidirectionalSemanticUnit, self).__init__()
        
        self.embedding = nn.Embedding(embedding_matrix.size(0),
                                      embedding_matrix.size(1),
                                      padding_idx=padding_idx)
        self.embedding.load_state_dict({'weight': embedding_matrix})
        self.embedding.weight.requires_grad = False

        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(batch_first=True, 
                            hidden_size=hidden_size, 
                            num_layers=2,
                            bidirectional=True,
                            dropout=.2)

        self.multimodal = MultiModalSemanticUnit(
            n_branches=2, 
            in_features=self.hidden_size, 
            out_features=out_features)

    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        batch, seq_len = x.size()
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        h_n = h_n[-1] # last layer

        # x: [N_LAYERS, DIRECTIONS, BATCH, HIDDEN_SIZE]
        x = h_n.view(-1, 2, batch, self.hidden_size)
        
        # Join hidden states from both directions
        return self.multimodal(x[-1])
