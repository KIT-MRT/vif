import torch
from torch import nn
from torchvision.models import (
    vit_b_16,
    ViT_B_16_Weights,
    vit_l_16,
    ViT_L_16_Weights,
)
import timm

from vit_pytorch import SimpleViT


def pooling_operation(pool_type, size):
    if pool_type == "avg":
        return nn.AvgPool2d(size)
    elif pool_type == "max":
        return nn.MaxPool2d(size)
    else:
        raise ValueError("Invalid pooling type. Use 'avg' or 'max'.")


class VitBackbone(nn.Module):
    def __init__(self, frozen, dim, pretrain):
        super(VitBackbone, self).__init__()

        self.frozen = frozen
        self.dim = dim

        if self.dim == 768 and not pretrain:
            self.image_backbone = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
            self.image_backbone.heads = nn.Sequential()
        if self.dim == 1024 and not pretrain:
            self.image_backbone = vit_l_16(weights=ViT_L_16_Weights.DEFAULT)
            self.image_backbone.heads = nn.Sequential()
        if pretrain:
            self.image_backbone = SimpleViT(
                image_size=(224, 224),
                patch_size=8,
                num_classes=1,
                dim=768,
                depth=6,
                heads=16,
                mlp_dim=768 * 4,
            )

            pretrained_ckpt = torch.load(pretrain)
            pretrained_dict = pretrained_ckpt["state_dict"]
            model_dict = self.image_backbone.state_dict()
            pretrained_dict = {
                k.replace("encoder.", "", 1): v for k, v in pretrained_dict.items()
            }
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items() if k in model_dict
            }
            model_dict.update(pretrained_dict)
            self.image_backbone.linear_head = nn.Sequential()

        if self.frozen:
            self.image_backbone.eval()

    def forward(self, images):
        b, i, c, h, w = images.shape

        if self.frozen:
            with torch.no_grad():
                image_encoding = self.image_backbone(images.view(b * i, c, h, w)).view(
                    b, i, self.dim
                )
        else:
            image_encoding = self.image_backbone(images.view(b * i, c, h, w)).view(
                b, i, self.dim
            )

        return image_encoding


class ConvNextBackbone(nn.Module):
    def __init__(self, transformer_dim, frozen=True) -> None:
        super(ConvNextBackbone, self).__init__()

        self.camera_backbone = timm.create_model(
            "convnextv2_base", pretrained=True, features_only=True
        )
        self.frozen = frozen
        if self.frozen:
            self.camera_backbone.eval()
        self.transformer_dim = transformer_dim

        self.feature_embedding_0 = nn.Sequential(
            nn.AvgPool2d(56),
            nn.Flatten(start_dim=1),
            nn.LayerNorm(128),
            nn.Linear(128, self.transformer_dim),
            nn.LayerNorm(self.transformer_dim),
        )

        self.feature_embedding_1 = nn.Sequential(
            nn.AvgPool2d(28),
            nn.Flatten(start_dim=1),
            nn.LayerNorm(256),
            nn.Linear(256, self.transformer_dim),
            nn.LayerNorm(self.transformer_dim),
        )

        self.feature_embedding_2 = nn.Sequential(
            nn.AvgPool2d(14),
            nn.Flatten(start_dim=1),
            nn.LayerNorm(512),
            nn.Linear(512, self.transformer_dim),
            nn.LayerNorm(self.transformer_dim),
        )

        self.feature_embedding_3 = nn.Sequential(
            nn.AvgPool2d(7),
            nn.Flatten(start_dim=1),
            nn.LayerNorm(1024),
            nn.Linear(1024, self.transformer_dim),
            nn.LayerNorm(self.transformer_dim),
        )

    def forward(self, images):
        b, i, c, h, w = images.shape

        if self.frozen:
            with torch.no_grad():
                image_encoding = self.camera_backbone(images.view(b * i, c, h, w))
        else:
            image_encoding = self.encode(images.view(b * i, c, h, w))

        outs_0 = self.feature_embedding_0(image_encoding[0])
        outs_1 = self.feature_embedding_1(image_encoding[1])
        outs_2 = self.feature_embedding_2(image_encoding[2])
        outs_3 = self.feature_embedding_3(image_encoding[3])

        return (
            torch.stack((outs_0, outs_1, outs_2, outs_3), axis=1)
            .max(axis=1)[0]
            .view(b, i, self.transformer_dim)
        )


class ResnetBackbone(nn.Module):
    def __init__(
        self,
        transformer_dim,
        frozen=True,
        feature_pool_op="avg",
        transformer_pool_op="max",
        b0=True,
        b1=True,
        b2=True,
        b3=True,
        b4=True,
    ) -> None:
        super(ResnetBackbone, self).__init__()

        self.feature_pool_op = feature_pool_op
        self.transformer_pool_op = transformer_pool_op
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
        self.b4 = b4

        self.camera_backbone = timm.create_model(
            "resnet50", pretrained=True, features_only=True
        )
        self.frozen = frozen
        if self.frozen:
            self.camera_backbone.eval()
        self.transformer_dim = transformer_dim

        if self.b0:
            self.feature_embedding_0 = nn.Sequential(
                pooling_operation(self.transformer_pool_op, 112),
                nn.Flatten(start_dim=1),
                nn.LayerNorm(64),
                nn.Linear(64, self.transformer_dim),
                nn.LayerNorm(self.transformer_dim),
            )

        if self.b1:
            self.feature_embedding_1 = nn.Sequential(
                pooling_operation(self.transformer_pool_op, 56),
                nn.Flatten(start_dim=1),
                nn.LayerNorm(256),
                nn.Linear(256, self.transformer_dim),
                nn.LayerNorm(self.transformer_dim),
            )

        if self.b2:
            self.feature_embedding_2 = nn.Sequential(
                pooling_operation(self.transformer_pool_op, 28),
                nn.Flatten(start_dim=1),
                nn.LayerNorm(512),
                nn.Linear(512, self.transformer_dim),
                nn.LayerNorm(self.transformer_dim),
            )

        if self.b3:
            self.feature_embedding_3 = nn.Sequential(
                pooling_operation(self.transformer_pool_op, 14),
                nn.Flatten(start_dim=1),
                nn.LayerNorm(1024),
                nn.Linear(1024, self.transformer_dim),
                nn.LayerNorm(self.transformer_dim),
            )

        if self.b4:
            self.feature_embedding_4 = nn.Sequential(
                pooling_operation(self.transformer_pool_op, 7),
                nn.Flatten(start_dim=1),
                nn.LayerNorm(2048),
                nn.Linear(2048, self.transformer_dim),
                nn.LayerNorm(self.transformer_dim),
            )

        self.feature_learning = nn.Linear(
            self.transformer_dim * 5, self.transformer_dim
        )

    def forward(self, images):
        b, i, c, h, w = images.shape

        if self.frozen:
            with torch.no_grad():
                image_encoding = self.camera_backbone(images.view(b * i, c, h, w))
        else:
            image_encoding = self.camera_backbone(images.view(b * i, c, h, w))

        outs = []
        if self.b0:
            outs.append(self.feature_embedding_0(image_encoding[0]))
        if self.b1:
            outs.append(self.feature_embedding_1(image_encoding[1]))
        if self.b2:
            outs.append(self.feature_embedding_2(image_encoding[2]))
        if self.b3:
            outs.append(self.feature_embedding_3(image_encoding[3]))
        if self.b4:
            outs.append(self.feature_embedding_4(image_encoding[4]))

        if self.transformer_pool_op == "max":
            return (
                torch.stack(outs, axis=1)
                .max(axis=1)[0]
                .view(b, i, self.transformer_dim)
            )

        if self.transformer_pool_op == "avg":
            return (
                torch.stack(outs, axis=1)
                .mean(axis=1)[0]
                .view(b, i, self.transformer_dim)
            )

        if self.transformer_pool_op == "learned":
            return self.feature_learning(torch.concat(outs, dim=1)).view(
                b, i, self.transformer_dim
            )

        raise ValueError(
            "Invalid transformer pooling type. Use 'avg', 'max', or 'learned'."
        )


class AttentionBackbone(nn.Module):
    def __init__(self, backbone) -> None:
        super(AttentionBackbone, self).__init__()

        self.attention = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                dilation=(1, 2),
                padding="same",
            ),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                dilation=(1, 2),
                padding="same",
            ),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                dilation=(1, 2),
                padding="same",
            ),
            nn.Conv2d(
                in_channels=64,
                out_channels=1,
                kernel_size=3,
                dilation=(1, 2),
                padding="same",
            ),
        )

        self.backbone = backbone

    def forward(self, images):
        b, i, c, h, w = images.shape

        attn = self.attention(images.view(b * i, c, h, w))
        image_attn = images * attn.view(b, i, 1, h, w)

        return self.backbone(image_attn)
