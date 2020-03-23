import torch.nn as nn


class ZeroShotModel(nn.Module):

    def __init__(self, visual_fe: nn.Module, semantic_unit: nn.Module):
        self.visual_fe = visual_fe 
        self.visual_fe.requires_grad = False

        self.semantic_unit = semantic_unit
        self.linear = nn.Linear(semantic_unit.out_features, 
                                self.visual_fe.features)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, 
                image: torch.FloatTensor, 
                semantic_repr: torch.Tensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        
        im_features = self.visual_fe(image)

        semantic_embedding = self.semantic_unit(semantic_repr)
        semantic_embedding = self.relu(self.linear(semantic_embedding))
        return im_features, semantic_embedding
