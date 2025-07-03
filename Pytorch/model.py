import torch.nn as nn
from S3D import s3d_linh, S3D_Weights

import yaml

def load_config(path="../Configurate/train.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()

in_channels = config["model"]["in_channels"]

def build_s3d_model(num_classes=15, in_channels = in_channels, pretrained=True, freeze_until_layer=10):
    if pretrained:
        weights = S3D_Weights.KINETICS400_V1
        model = s3d_linh(weights=weights)
    else:
        model = s3d_linh(weights=None)


    model.classifier[1] = nn.Conv3d(in_channels, num_classes, kernel_size=1, stride=1, bias=True)

    for idx, (name, param) in enumerate(model.features.named_parameters()):
        if idx < freeze_until_layer:
            param.requires_grad = False

    return model
