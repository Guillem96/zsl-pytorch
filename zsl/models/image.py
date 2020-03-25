import torch
import torch.nn as nn
import torchvision.models as zoo


_AVAILABLE_FE = ['resnet50', 'resnet152', 'densenet121', 'inception_v3']


class VisualFeatureExtractor(nn.Module):

    def __init__(self, feature_extractor: str, trainable: bool = False):
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

        if feature_extractor.startswith('resnet'):
            self.conv, self.features = \
                _build_resnet(getattr(zoo, feature_extractor)(pretrained=True))
        elif feature_extractor == 'densenet121':
            self.conv, self.features = \
                _build_densenet(zoo.densenet121(pretrained=True))
        elif feature_extractor == 'inception_v3':
            inception = zoo.inception_v3(pretrained=True)
            self.conv = nn.Sequential(
                inception.Conv2d_1a_3x3,
                inception.Conv2d_2a_3x3, inception.Conv2d_2b_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2),

                inception.Conv2d_3b_1x1, 

                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2),

                inception.Mixed_5b, inception.Mixed_5c, inception.Mixed_5d,

                inception.Mixed_6a, inception.Mixed_6b, inception.Mixed_6c,
                inception.Mixed_6d, inception.Mixed_6e,

                inception.Mixed_7a, inception.Mixed_7b, inception.Mixed_7c,
                
                nn.AdaptiveAvgPool2d((1, 1)))
            self.features = inception.fc.in_features
            
        for p in self.conv.parameters():
            p.requires_grad = trainable

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.conv(x).view(x.size(0), -1)