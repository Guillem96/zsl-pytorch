import torch
import torch.nn as nn
import torchvision.models as zoo


_AVAILABLE_FE = ['resnet50']


class VisualFeatureExtractor(nn.Module):

    def __init__(self, feature_extractor: str):
        super(VisualFeatureExtractor, self).__init__()

        assert feature_extractor in _AVAILABLE_FE, \
            f'{feature_extractor} is not available'
        
        def _build_resnet(resnet):
            return nn.Sequential(
                resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
                nn.AdaptiveAvgPool2d((1, 1))), resnet.fc.in_features

        def _build_densenet(densenet):
            return nn.Sequential(
                densenet.features, nn.ReLU(inplace=True), 
                nn.AdaptiveAvgPool2d((1, 1))), densenet.classifier.in_features

        if feature_extractor == 'resnet50':
            self.conv, self.features = \
                _build_resnet(zoo.resnet50(pretrained=True))
        elif feature_extractor == 'densenet121':
            self.conv, self.features = \
                _build_densenet(zoo.densenet121(pretrained=True))
    
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.conv(x).view(x.size(0), -1)