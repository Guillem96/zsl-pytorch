import sys
from typing import Sequence, Callable, Mapping

import torch
import torch.nn.functional as F

from .models import ZeroShot
from .metrics import accuracy, top_5_accuracy


def train_epoch(model: ZeroShot, 
                dl: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer,
                epoch: int,
                print_freq: int = 20,
                device: torch.device = torch.device('cpu')):
    """
    Train the model for an epoch. The model will see all the data once

    Parameters
    ----------
    model: zsl.models.ZeroShot
        Model that returns a tuple containing the visual and semantic embedding
    dl: torch.utils.data.DataLoader
        DataLoader to iterate batch of images. The collate function of the 
        dataloader must return a tuple of three elements 
        [images, labels, semantic label encoding]
    optimizer: torch.optim.Optimizer
        Optimizer to update the model weights
    epoch: int
        Current epoch
    print_freq: int, default 20
        Report training every `print_freq` iterations
    device: torch.device, default torch.device('cpu')
        torch.device('cuda') or torch.device('cpu')
    """
    running_loss = 0.
    for i, (images, _, semantics) in enumerate(dl):
        model.train()

        print('.', end='')
        sys.stdout.flush()

        images = images.to(device)
        semantics = semantics.to(device)

        visual_embeds, semantic_embed = model(images, semantics)

        # Get visual embeds and semantic embeds as closer as possible
        loss = F.mse_loss(visual_embeds, semantic_embed)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 5.)
        optimizer.step()

        running_loss += loss.item()
        
        if (i + 1) % print_freq == 0:
            loss_mean = running_loss / i
            print()
            print(f'Epoch [{epoch}] [{i}/{len(dl)}] '
                  f'loss: {loss_mean:.4f}')

    loss_mean = running_loss / len(dl)
    print()
    print(f'Epoch [{epoch}] [{len(dl)}/{len(dl)}] loss: {loss_mean:.4f}')

_Metric = Callable[[torch.FloatTensor, 
                    torch.FloatTensor, 
                    torch.LongTensor], torch.FloatTensor]

@torch.no_grad()
def evaluate(model: ZeroShot,
             dl: torch.utils.data.DataLoader,
             class_representations: Sequence[torch.Tensor],
             metrics: Mapping[str, _Metric] = {'accuracy': accuracy, 
                                               'top_5_accuracy': top_5_accuracy},
             device: torch.device = torch.device('cpu'),
             padding_idx: int = 0):
    """
    Function to evaluate the model on the whole validation set.

    Parameters
    ----------
    model: zsl.models.ZeroShot
        Model that returns a tuple containing the visual and semantic embedding
    dl: torch.utils.data.DataLoader
        DataLoader to iterate batch of images. The collate function of the 
        dataloader must return a tuple of three elements 
        [images, labels, semantic label encoding]
    class_representations: Sequence[torch.Tensor]
        Ordered semantic representations of the labels. Ordered means that the 
        representation of label n should be on the position n.
    metrics: Mapping[_Metric]
        Mapping of name to function that receives three parameters: 
        [visual_embedding, class_representation, labels] and returns an 
        score
    device: torch.device, default torch.device('cpu')
        torch.device('cuda') or torch.device('cpu')
    """
    model.eval()

    # Convert descriptions to tensor padding it
    descriptions = [torch.tensor(d) if not isinstance(d, torch.Tensor) else d
                    for d in class_representations]
    descriptions = torch.nn.utils.rnn.pad_sequence(
        descriptions, batch_first=True, padding_value=padding_idx)
    semantic_repr = model(semantic_repr=descriptions.to(device))

    # Initialize running metrics to 0
    running_metrics = {m: 0. for m in metrics}
    running_metrics['loss'] = 0.

    for i, (images, y_true, semantics) in enumerate(dl):
        print('.', end='')
        sys.stdout.flush()

        images = images.to(device)
        semantics = semantics.to(device)
        y_true = y_true.to(device)

        visual_embeds, semantic_embed = model(images, semantics)
        loss = F.mse_loss(visual_embeds, semantic_embed)

        running_metrics['loss'] += loss.item()
        for m, f in metrics.items():
            running_metrics[m] += f(visual_embeds, semantic_repr, 
                                    y_true).item()

    mean_metrics = {m: v / float(len(dl)) for m, v in running_metrics.items()}
    metrics_str = ' '.join(f'{m}: {v:.4f}' for m, v in mean_metrics.items())

    print()
    print('Validation', metrics_str)
    
