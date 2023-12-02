import argparse

from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from torchvision import transforms
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from model.baseline.baseline_3d import generate_model
from model.baseline.deepsignals import DeepsignalsBaseline
from model.vif_create import (
    AttnBackboneConfig,
    ResnetBackboneConfig,
    VifConfig,
    VitBackboneConfig,
    create_vif,
)
from model.lightning import LitVisualIntentionFormer
from utils.augmentations import augment_sequences
from utils.dataset import VisualIntentionsDictDataset, create_data_dict
from utils.logging import TbImageLogger


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_annotations", type=str)
    parser.add_argument("val_annotations", type=str)
    parser.add_argument("train_images_dir", type=str)
    parser.add_argument("val_images_dir", type=str)
    parser.add_argument("method", type=str, default="vif")
    parser.add_argument("--sequence_len", type=int, default=20, help="sequence length")
    parser.add_argument("--num_gpus", type=int, default=1, help="Num GPUs (default: 1)")
    parser.add_argument(
        "--num_nodes", type=int, default=1, help="Num nodes (default: 1)"
    )
    parser.add_argument("--train_id", type=int, default=0, help="")
    args = parser.parse_args()

    train_annotations = args.train_annotations
    val_annotations = args.val_annotations
    train_images_dir = args.train_images_dir
    val_images_dir = args.val_images_dir
    method = args.method
    sequence_len = args.sequence_len
    num_gpus = args.num_gpus
    num_nodes = args.num_nodes
    train_id = args.train_id

    TRAIN_HOURS = 1
    BATCH_SIZE = 2*num_gpus
    MIN_IMAGE_WIDTH = 120
    EPOCHS = 10
    START_LR = 1e-5
    END_LR = 1e-6
    GAMMA = 0.7
    MAX_OCC = 1.0
    print("create train data dict")
    train_data_dict = create_data_dict(
        train_annotations,
        train_images_dir,
        sequence_len=sequence_len,
        min_image_width=MIN_IMAGE_WIDTH,
        max_occupancy=MAX_OCC,
        cache_file=f"train_{MAX_OCC}occ_{sequence_len}sl.pkl",
        only_intentions=True,
    )
    print("create val data dict")
    val_data_dict = create_data_dict(
        val_annotations,
        val_images_dir,
        sequence_len=sequence_len,
        min_image_width=MIN_IMAGE_WIDTH,
        max_occupancy=MAX_OCC,
        cache_file=f"val_{MAX_OCC}occ_{sequence_len}sl.pkl",
        only_intentions=True,
    )

    mean = [0.27678646, 0.30232353, 0.34076891]
    std = [0.13243662, 0.13241449, 0.14208872]
    normalize = transforms.Normalize(
        mean=mean,
        std=std,
    )
    inv_normalize = transforms.Normalize(
        mean=[-mean[0] / std[0], -mean[1] / std[1], -mean[2] / std[2]],
        std=[1 / std[0], 1 / std[1], 1 / std[2]],
    )

    training_dataset = VisualIntentionsDictDataset(
        train_data_dict,
        image_transform=transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    train_dataloader = DataLoader(
        training_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        collate_fn=augment_sequences,
    )

    validation_dataset = VisualIntentionsDictDataset(
        val_data_dict,
        image_transform=transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    val_dataloader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=24,
    )

    NUM_IND_CLASSES = 4
    WITH_HEADING_FEATURE = True
    if method == "vif":
        DIM = 1024
        model_config = VifConfig(
            sequence_len=sequence_len,
            min_img_width=MIN_IMAGE_WIDTH,
            backbone=VitBackboneConfig(
                frozen=False,
                # pretrain="/code/checkpoints/pretrain_8x8.ckpt",
                dim=DIM,
            ),
            dim=DIM,
            depth=2,
            heads=16,
            mlp_dim=DIM * 4,
            dim_head=64,
            attn_dropout=0.75,  # 0.75
            head_dropout=0.75,  # 0.75
            transformer_dropput=0.0,  # 0.75
            with_rear=True,
            with_heading=True,
            with_cls=True,
            num_ind_classes=NUM_IND_CLASSES,
            with_heading_feature=WITH_HEADING_FEATURE,
        )
        print(model_config)
        model = create_vif(config=model_config)
        logger = TensorBoardLogger("tb_logs", name=model_config.name())
    elif method == "3dresnet":
        model = generate_model(101, num_ind_classes=NUM_IND_CLASSES)
        logger = TensorBoardLogger("tb_logs", name="3dresnet")
    elif method == "deepsignals":
        model = DeepsignalsBaseline(num_ind_classes=NUM_IND_CLASSES)
        logger = TensorBoardLogger("tb_logs", name="deepsignals")
    else:
        import sys

        sys.exit(-1)

    # img_val_logger = TbImageLogger(logger, inv_normalize)
    # img_train_logger = TbImageLogger(logger, inv_normalize)
    lit = LitVisualIntentionFormer(
        model, START_LR, END_LR, num_ind_classes=NUM_IND_CLASSES, max_epochs=EPOCHS
    )

    print("start training")
    checkpoint_callback = ModelCheckpoint(
        save_top_k=5,
        monitor="val_ind_f1",
        mode="max",
        dirpath="checkpoints/",
        filename=f"{method}_{sequence_len}_{train_id}_{int(WITH_HEADING_FEATURE)}"
        + "_{epoch:02d}-{val_ind_f1:.3f}",
    )

    trainer = L.Trainer(
        precision="16-mixed",
        accelerator="gpu",
        devices=num_gpus,
        num_nodes=num_nodes,
        strategy="ddp_find_unused_parameters_true",
        max_time={"days": 0, "hours": TRAIN_HOURS},
        max_epochs=EPOCHS,
        check_val_every_n_epoch=1,
        logger=logger,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(lit, train_dataloader, val_dataloader)
