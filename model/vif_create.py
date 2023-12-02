from dataclasses import dataclass

from model.image_backbone import (
    AttentionBackbone,
    ResnetBackbone,
    VitBackbone,
)
from model.vif import VisualIntentionFormer


@dataclass
class BackboneConfig:
    frozen: bool


@dataclass
class ResnetBackboneConfig(BackboneConfig):
    out_dim: int
    feature_pool_op: str
    transformer_pool_op: str
    b0: bool = True
    b1: bool = True
    b2: bool = True
    b3: bool = True
    b4: bool = True


@dataclass
class AttnBackboneConfig(BackboneConfig):
    backbone: BackboneConfig


@dataclass
class VitBackboneConfig(BackboneConfig):
    dim: int = None
    pretrain: str = None


@dataclass
class VifConfig:
    sequence_len: int
    min_img_width: int
    backbone: BackboneConfig
    dim: int
    depth: int
    heads: int
    mlp_dim: int
    dim_head: int
    with_rear: bool
    with_heading: bool
    with_cls: bool
    attn_dropout: float
    head_dropout: float
    transformer_dropput: float
    num_ind_classes: int
    with_heading_feature: bool

    def name(self):
        return f"{self.sequence_len}_{self.backbone}_{self.dim},{self.depth},{self.heads},{self.mlp_dim},{self.dim_head}_{int(self.with_rear)}{int(self.with_heading)}_{self.with_cls}_{self.attn_dropout},{self.head_dropout},{self.transformer_dropput}_ind{self.num_ind_classes}_{self.with_heading_feature}"


def get_backbone(config: BackboneConfig):
    image_backbone = None
    if isinstance(config, ResnetBackboneConfig):
        return ResnetBackbone(config.out_dim, frozen=config.frozen)

    if isinstance(config, VitBackboneConfig):
        return VitBackbone(config.frozen, config.dim, config.pretrain)

    if isinstance(config, AttnBackboneConfig):
        backbone = get_backbone(config.backbone)
        return AttentionBackbone(backbone)

    if image_backbone is None:
        raise ValueError(f"unknown backbone {config}")


def create_vif(config: VifConfig):
    image_backbone = get_backbone(config.backbone)

    return VisualIntentionFormer(
        image_backbone=image_backbone,
        depth=config.depth,
        heads=config.heads,
        dim=config.dim,
        mlp_dim=config.mlp_dim,
        dim_head=config.dim_head,
        with_rear=config.with_rear,
        with_heading=config.with_heading,
        with_cls=config.with_cls,
        attn_dropout=config.attn_dropout,
        head_dropout=config.head_dropout,
        transformer_dropout=config.transformer_dropput,
        num_ind_classes=config.num_ind_classes,
        with_heading_feature=config.with_heading_feature,
    )
