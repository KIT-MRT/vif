import torch
from torch import nn
import einops
from einops import rearrange

from model.simple_vit import Transformer


def tempembed_sincos_1d(images, temperature=1000, dtype=torch.float32):
    _, n, dim, device, dtype = *images.shape, images.device, images.dtype

    n = torch.arange(n, device=device)
    assert (dim % 2) == 0, "feature dimension must be multiple of 2 for sincos emb"
    omega = torch.arange(dim // 2, device=device) / (dim // 2 - 1)
    omega = 1.0 / (temperature**omega)

    n = n.flatten()[:, None] * omega[None, :]
    te = torch.cat((n.sin(), n.cos()), dim=1)
    return te.type(dtype)


class VisualIntentionFormer(nn.Module):
    def __init__(
        self,
        image_backbone,
        depth,
        heads,
        dim,
        mlp_dim,
        dim_head,
        with_heading=True,
        with_rear=True,
        with_cls=True,
        with_heading_feature=False,
        attn_dropout=0.0,
        head_dropout=0.0,
        transformer_dropout=0.0,
        num_ind_classes=4,
    ):
        super(VisualIntentionFormer, self).__init__()

        # Image Backbone
        self.image_backbone = image_backbone

        #
        # Transformer Parameters
        #
        self.depth = depth
        self.heads = heads
        self.dim = dim
        self.mlp_dim = mlp_dim
        self.dim_head = dim_head

        self.with_rear = with_rear
        self.with_heading = with_heading
        self.with_cls = with_cls

        self.attn_dropout = attn_dropout
        self.head_dropout = head_dropout
        self.transformer_dropout = transformer_dropout

        if with_cls:
            self.cls_token_indication = nn.Parameter(torch.randn(1, 1, self.dim))

        # self.cls_token_rear = nn.Parameter(torch.randn(1, 1, self.dim))

        # Transformer Encoder
        self.transformer = Transformer(
            self.dim, self.depth, self.heads, self.dim_head, self.mlp_dim
        )
        self.to_latent = nn.Identity()

        # Intention Classification Head
        self.dropout = torch.nn.Dropout(self.transformer_dropout)

        self.with_heading_feature = with_heading_feature
        indicator_dim = self.dim + int(with_heading_feature)
        self.indicator_head = nn.Sequential(
            nn.LayerNorm(indicator_dim),
            nn.Linear(indicator_dim, num_ind_classes),  # None, Left, Right, Emergency
        )

        if self.with_rear:
            self.rear_head = nn.Sequential(
                nn.LayerNorm(self.dim), nn.Linear(self.dim, 3)  # None, Rear, Break
            )

        if self.with_heading:
            self.heading_head = nn.Sequential(
                nn.LayerNorm(self.dim),
                nn.Linear(self.dim, 4),  # Front, Left, Right, Back
            )

    def forward(self, images, headings):
        b, i, c, h, w = images.shape

        #
        # Image Feature Extraction
        #
        image_embedding = self.image_backbone(images)

        #
        # Sequence Extraction
        #

        # [CLS] Token
        if self.with_cls:
            cls_token_indication = einops.repeat(
                self.cls_token_indication, "1 1 d -> b 1 d", b=b
            )

        # Transformer
        te = tempembed_sincos_1d(image_embedding)
        image_embedding = image_embedding + te
        if self.with_cls:
            image_embedding = torch.cat((cls_token_indication, image_embedding), dim=1)
        sequence_encoding = self.transformer(image_embedding)

        # Output
        if self.with_cls:
            sequence_encoding = sequence_encoding[:, 0]
        else:
            sequence_encoding = sequence_encoding.mean(dim=1)
        sequence_encoding = self.dropout(sequence_encoding)

        #
        # Classification Heads
        #
        sequence_encoding_indicator = sequence_encoding
        if self.with_heading_feature:
            sequence_encoding_indicator = torch.cat(
                (sequence_encoding, headings[:, -1].unsqueeze(1)), dim=1
            )

        pred_indicator = self.indicator_head(sequence_encoding_indicator)
        preds = []
        preds.append(pred_indicator)
        if self.with_rear:
            preds.append(self.rear_head(sequence_encoding))

        if self.with_heading:
            preds.append(self.heading_head(sequence_encoding))

        return preds
