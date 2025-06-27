import torch.nn as nn
from S3D import s3d_linh, S3D_Weights

def build_s3d_model(num_classes=15, pretrained=True, freeze_until_layer=10):
    if pretrained:
        weights = S3D_Weights.KINETICS400_V1
        model = s3d_linh(weights=weights)
    else:
        model = s3d_linh(weights=None)


    in_channels = model.classifier[1].in_channels
    model.classifier[1] = nn.Conv3d(in_channels, num_classes, kernel_size=1, stride=1, bias=True)

    for idx, (name, param) in enumerate(model.features.named_parameters()):
        if idx < freeze_until_layer:
            param.requires_grad = False

    return model
