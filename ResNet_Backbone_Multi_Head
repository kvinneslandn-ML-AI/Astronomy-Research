# ============================================================
# ResNet backbone + multitask heads (1-channel input)
# ============================================================
from torchvision import models
import torch.nn.functional as F

class ResNetBackbone(nn.Module):
    def __init__(self, name="resnet18", in_channels=1, pretrained=False):
        super().__init__()

        if name == "resnet18":
            net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            feat_dim = 512
        elif name == "resnet34":
            net = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
            feat_dim = 512
        elif name == "resnet50":
            net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            feat_dim = 2048
        else:
            raise ValueError("Use resnet18 or resnet34 or resnet50")

        # replace first conv to accept 1-channel input
        net.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        self.body = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool,
            net.layer1, net.layer2, net.layer3, net.layer4
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feat_dim = feat_dim

    def forward(self, x):
        x = self.body(x)          # (B,C,h,w)
        x = self.pool(x)          # (B,C,1,1)
        return x.flatten(1)       # (B,C)


class LensRadiusResNet(nn.Module):
    def __init__(self, backbone="resnet18", in_channels=1, hidden=256, pretrained=False):
        super().__init__()
        self.backbone = ResNetBackbone(backbone, in_channels, pretrained)

        self.shared = nn.Sequential(
            nn.Linear(self.backbone.feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, hidden),
            nn.ReLU(inplace=True),
        )
        self.cls_head = nn.Linear(hidden, 1)
        self.rad_head = nn.Linear(hidden, 1)

    def forward(self, x):
        f = self.backbone(x)
        h = self.shared(f)
        lens_logit = self.cls_head(h).squeeze(1)
        radius     = F.softplus(self.rad_head(h)).squeeze(1)
        return lens_logit, radius
