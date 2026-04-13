import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from models.cbam import CBAM

class SiameseChangeNet(nn.Module):

    def __init__(self):
        super(SiameseChangeNet, self).__init__()

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        self.encoder = nn.Sequential(*list(resnet.children())[:-2])

        self.cbam = CBAM(512)

        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        print("MODEL RUNNING")  # 🔥 DEBUG LINE

        f1 = self.encoder(x1)
        f2 = self.encoder(x2)

        f1 = self.cbam(f1)
        f2 = self.cbam(f2)

        diff = torch.abs(f1 - f2)

        out = self.decoder(diff)

        out = F.interpolate(out, size=(256, 256), mode='bilinear', align_corners=False)

        return out
