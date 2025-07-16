import math
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.ops import SqueezeExcitation

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]
class ecalayer(nn.Module):
    def __init__(self, channel, k_size=3):
        super(ecalayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

def replace_mbconv_with_ghost(model, indices=[3]):
    for i in indices:
        layer = model.features[i]
        if not isinstance(layer, nn.Sequential) or len(layer) == 0:
            continue

        mbconv = layer[0]
        if not hasattr(mbconv, "block"):
            continue

        block_seq = mbconv.block
        in_channels = block_seq[0][0].in_channels
        out_channels = block_seq[-1][0].out_channels

        ghost = GhostModule(
            inp=in_channels,
            oup=out_channels,
            kernel_size=3,
            stride=1,
            ratio=2,
            dw_size=3,
            relu=True
        )
        model.features[i] = ghost
    return model

def replace_se_with_eca(model, k_size=3):
    for name, module in model.named_children():
        if isinstance(module, SqueezeExcitation):
            in_channels = module.fc1.in_channels
            new_module = ecalayer(channel=in_channels, k_size=k_size)
            setattr(model, name, new_module)
        else:
            replace_se_with_eca(module, k_size=k_size)
    return model

def efficientnet(num_classes=2):
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    model = replace_mbconv_with_ghost(model, indices=[3])  
    model = replace_se_with_eca(model, k_size=3)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model
