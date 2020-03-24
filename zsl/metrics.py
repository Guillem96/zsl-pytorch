from functools import partial

import torch


def l2_dist(a: torch.FloatTensor, b: torch.FloatTensor) -> torch.FloatTensor:
    '''
    Credits: https://discuss.pytorch.org/t/batched-pairwise-distance/39611/2
    
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
    a_norm = (a**2).sum(1).view(-1, 1)

    b_t = torch.transpose(b, 0, 1)
    b_norm = (b**2).sum(1).view(1, -1)
    
    dist = a_norm + b_norm - 2.0 * torch.mm(a, b_t)
    return dist.clamp(min=0.0)


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
    
    preds_k = (-dists).topk(k).indices
    y_true = y_true.unsqueeze(1).repeat(1, k)

    pred_ids = predict(image_embeddigs, class_semantics)
    return (preds_k == y_true).any(-1).sum() / float(y_true.size(0))


top_5_accuracy = partial(top_k_accuracy, k=5)
top_3_accuracy = partial(top_k_accuracy, k=3)
accuracy = partial(top_k_accuracy, k=1)
