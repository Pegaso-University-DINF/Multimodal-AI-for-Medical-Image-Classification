import torch.nn as nn
import torchvision.models as models

def get_resnet_model(model_name='resnet50', num_classes=2, pretrained=True):
    """Return a ResNet model with customized final layer"""
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
    elif model_name == 'resnet34':
        model = models.resnet34(pretrained=pretrained)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
    elif model_name == 'resnet101':
        model = models.resnet101(pretrained=pretrained)
    else:
        raise ValueError("Model name must be one of ['resnet18','resnet34','resnet50','resnet101']")

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
