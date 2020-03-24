from functools import partial

import torch


def l2_dist(a: torch.FloatTensor, b: torch.FloatTensor) -> torch.FloatTensor:
    '''
    Implements L2 Distance

    Parameters
    ----------
    a: torch.FloatTensor
        [N, D] Tensor
    b: torch.FloatTensor
        [M, D] Tensor
    
    Returns
    -------
    torch.Tensor
        [N, M] Tensor where index i,j contains the distance between 
        a[i] and y[j]
    '''
    dists = torch.ones(a.size(0), b.size(0)).to(a.device)
    for i in range(a.size(0)):
        dists[i] = (a[i] - b).pow(2).sum(dim=-1).sqrt()

    return dists.clamp(min=0.0)


def top_k_accuracy(image_embeddigs: torch.FloatTensor, 
                   class_semantics: torch.FloatTensor,
                   y_true: torch.LongTensor,
                   k: int = 5) -> float:
    """
    Computes the top k accuracy
    """
    y_true = y_true.view(-1)

    assert image_embeddigs.size(0) == y_true.size(0)
    
    dists = l2_dist(image_embeddigs, class_semantics)
    
    preds_k = (-dists).topk(k, dim=-1).indices
    y_true = y_true.unsqueeze(1).repeat(1, k)

    return (preds_k == y_true).any(-1).sum() / float(y_true.size(0))


top_5_accuracy = partial(top_k_accuracy, k=5)
top_3_accuracy = partial(top_k_accuracy, k=3)
accuracy = partial(top_k_accuracy, k=1)
