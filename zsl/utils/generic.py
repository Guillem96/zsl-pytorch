import random
from typing import Sequence, Any, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

import numpy as np


def collate_image_folder(batch: Tuple[Any, Any, Any], 
                         padding_idx: int = -1) -> Tuple[torch.FloatTensor,
                                                         torch.LongTensor, 
                                                         torch.Tensor]:
    """
    Predefinet collate function for torch.DataLoader.

    Parameters
    ----------
    batch: Tuple[Any, _, Any]
    padding_idx: int, default -1
        Padding value used to pad the semantic representation of labels
    
    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
    """
    images, labels, semantics = zip(*batch)
    semantics = [o if isinstance(o, torch.Tensor) else torch.tensor(o) 
                 for o in semantics]
    semantics = pad_sequence(semantics, batch_first=True)
    return torch.stack(images), torch.LongTensor(labels), semantics

    
def seed(s: int = 0):
    """
    Sets the random seed to zero for all libraries including "randomness"

    - random
    - numpy
    - torch
    
    """
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def split_classes(all_classes: Sequence[Any], 
                  zs_prob: float) -> Tuple[Sequence[Any], Sequence[Any]]:
    """
    Splits a sequence of elements in two sets. 
    This function is handy when splitting classes in zero shot and
    non sero shot.

    Parameters
    ----------
    all_classes: Sequence[Any]
        All classes. 
    zs_prob: float
        Pobability that a class belongs to zero shot learning
    
    Retruns
    -------
    Tuple[Sequenc[Any], Sequence[Any]]

    """
    zs_classes = []
    classes = []

    for c in all_classes:
        if random.random() < zs_prob:
            zs_classes.append(c)
        else: 
            classes.append(c)
    
    return classes, zs_classes
