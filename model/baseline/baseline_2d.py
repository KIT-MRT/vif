import torch
from torch import nn
import timm


class ResnetBaseline(nn.Module):
    def __init__(self, past_frames):
        super().__init__()
        self.resnet = timm.create_model(
            "resnet101",
            pretrained=True,
            in_chans=past_frames * 3,
            num_classes=5,
            drop_rate=0.3,
        )

    def forward(self, images):
        b, i, c, h, w = images.shape

        output = self.resnet(images.view(b, i * c, h, w))
        return output
