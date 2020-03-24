from typing import Tuple

import torch
import torch.nn as nn


class ZeroShot(nn.Module):

    def __init__(self, visual_fe: nn.Module, semantic_unit: nn.Module):
        super(ZeroShot, self).__init__()

        self.visual_fe = visual_fe 
        for p in self.visual_fe.parameters():
            p.requires_grad = False

        self.semantic_unit = semantic_unit
        self.linear = nn.Linear(semantic_unit.out_features, 
                                self.visual_fe.features)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, 
                image: torch.FloatTensor = None, 
                semantic_repr: torch.Tensor = None):
        
        res = []
        if image is not None:
            im_features = self.visual_fe(image)
            res.append(im_features)
        
        if semantic_repr is not None:
            semantic_embedding = self.semantic_unit(semantic_repr)
            semantic_embedding = self.relu(self.linear(semantic_embedding))
            res.append(semantic_embedding)

        return res
