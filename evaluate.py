import argparse

import numpy as np
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm

from model.baseline.baseline_3d import generate_model
from model.baseline.deepsignals import DeepsignalsBaseline
from model.vif_create import VifConfig, VitBackboneConfig, create_vif
from model.lightning import LitVisualIntentionFormer
from utils.dataset import VisualIntentionsDictDataset, create_data_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("test_annotations", type=str)
    parser.add_argument("test_images_dir", type=str)
    parser.add_argument("model_ckpt", type=str)
    args = parser.parse_args()

    test_annotations = args.test_annotations
    test_images_dir = args.test_images_dir
    model_ckpt = args.model_ckpt
    mean = [0.27678646, 0.30232353, 0.34076891]
    std = [0.13243662, 0.13241449, 0.14208872]
    normalize = transforms.Normalize(
        mean=mean,
        std=std,
    )

    NUM_IND_CLASSES = 4
    SEQUENCE_LEN = 10
    MIN_IMAGE_WIDTH = 120
    DIM = 1024
    model_config = VifConfig(
        sequence_len=SEQUENCE_LEN,
        min_img_width=MIN_IMAGE_WIDTH,
        backbone=VitBackboneConfig(
            frozen=False,
            dim=DIM,
        ),
        dim=DIM,
        depth=2,
        heads=16,
        mlp_dim=DIM * 4,
        dim_head=64,
        attn_dropout=0.75,
        head_dropout=0.75,
        transformer_dropput=0.0,
        with_rear=True,
        with_heading=True,
        with_cls=True,
        num_ind_classes=NUM_IND_CLASSES,
        with_heading_feature=True,
    )
    print(model_config)
    model = create_vif(config=model_config)
    # model = generate_model(101, num_ind_classes=num_ind_classes)
    # model = DeepsignalsBaseline(num_ind_classes=num_ind_classes)

    model = LitVisualIntentionFormer(model, 0, 0, num_ind_classes=NUM_IND_CLASSES)
    checkpoint = torch.load(model_ckpt)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.vif
    model.to("cuda:0")
    model.eval()

    test_data_dict = create_data_dict(
        test_annotations,
        test_images_dir,
        sequence_len=SEQUENCE_LEN,
        min_image_width=MIN_IMAGE_WIDTH,
        max_occupancy=1.0,
        only_intentions=True
    )
    test_dataset = VisualIntentionsDictDataset(
        test_data_dict,
        image_transform=transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                normalize,
            ]
        ),
        with_context=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=8,
    )

    _preds_indicator = []
    _targets_indicator = []
    _preds_rear = []
    _targets_rear = []
    _preds_heading = []
    _targets_heading = []
    _weather = []
    _time = []
    for images, headings, targets, weather, time in tqdm(test_dataloader):
        images, headings, targets = (
            images.to("cuda:0"),
            headings.to("cuda:0"),
            targets.to("cuda:0"),
        )
        with torch.no_grad():
            pred_indicator, pred_rear, pred_heading = model(images, headings)

        if NUM_IND_CLASSES == 3:
            targets_indicator = targets[:, 1:4]
        else:
            targets_indicator = targets[:, :4]
        targets_rear = targets[:, 4:7]
        targets_heading = targets[:, 7:11]

        _preds_indicator.extend(pred_indicator.softmax(1).argmax(1).cpu())
        _targets_indicator.extend(targets_indicator.argmax(1).cpu())
        _preds_rear.extend(pred_rear.softmax(1).argmax(1).cpu())
        _targets_rear.extend(targets_rear.argmax(1).cpu())
        _preds_heading.extend(pred_heading.softmax(1).argmax(1).cpu())
        _targets_heading.extend(targets_heading.argmax(1).cpu())
        _weather.extend(weather)
        _time.extend(time)
    #
    # Total
    #
    f1_ind = f1_score(_targets_indicator, _preds_indicator, average="macro")
    p_ind = precision_score(_targets_indicator, _preds_indicator, average="macro")
    r_ind = recall_score(_targets_indicator, _preds_indicator, average="macro")

    f1_rear = f1_score(_targets_rear, _preds_rear, average="macro")
    p_rear = precision_score(_targets_rear, _preds_rear, average="macro")
    r_rear = recall_score(_targets_rear, _preds_rear, average="macro")

    f1_head = f1_score(_targets_heading, _preds_heading, average="macro")
    p_head = precision_score(_targets_heading, _preds_heading, average="macro")
    r_head = recall_score(_targets_heading, _preds_heading, average="macro")

    print("Total results:")
    print("-----------------------")
    print("f1_ind & f1_rear & f1_rear")
    print(f"{f1_ind} & {f1_rear} & {f1_head} \\")

    cm_ind = confusion_matrix(_targets_indicator, _preds_indicator)
    print(cm_ind)
    cm_ind = confusion_matrix(_targets_indicator, _preds_indicator, normalize="true")
    print(cm_ind)

    #
    # Heading
    #
    print("Heading based results:")
    print("-----------------------")
    _targets_indicator = np.array(_targets_indicator)
    _targets_rear = np.array(_targets_rear)
    _preds_indicator = np.array(_preds_indicator)
    _preds_rear = np.array(_preds_rear)
    _preds_heading = np.array(_preds_heading)
    _targets_heading = np.array(_targets_heading)
    idx_0 = _targets_heading == 0
    idx_1 = _targets_heading == 1
    idx_2 = _targets_heading == 2
    idx_3 = _targets_heading == 3
    f1_ind_0 = f1_score(
        _targets_indicator[idx_0], _preds_indicator[idx_0], average="macro"
    )
    f1_ind_1 = f1_score(
        _targets_indicator[idx_1], _preds_indicator[idx_1], average="macro"
    )
    f1_ind_2 = f1_score(
        _targets_indicator[idx_2], _preds_indicator[idx_2], average="macro"
    )
    f1_ind_3 = f1_score(
        _targets_indicator[idx_3], _preds_indicator[idx_3], average="macro"
    )
    print("Indicator")
    print("back, right, front, left")
    print(f"{f1_ind_0} & {f1_ind_1} & {f1_ind_2} & {f1_ind_3} \\")
    f1_ind_0 = f1_score(_targets_rear[idx_0], _preds_rear[idx_0], average="macro")
    f1_ind_1 = f1_score(_targets_rear[idx_1], _preds_rear[idx_1], average="macro")
    f1_ind_2 = f1_score(_targets_rear[idx_2], _preds_rear[idx_2], average="macro")
    f1_ind_3 = f1_score(_targets_rear[idx_3], _preds_rear[idx_3], average="macro")
    print("Rear")
    print("back, right, front, left")
    print(f"{f1_ind_0} & {f1_ind_1} & {f1_ind_2} & {f1_ind_3} \\")

    #
    # Time of Day
    #
    print("Time of Day based results:")
    print("-----------------------")
    _time = np.array(_time)
    idx_day = _time == "Day"
    idx_night = _time == "Night"
    idx_dawn_dusk = _time == "Dawn/Dusk"
    f1_ind_day = f1_score(
        _targets_indicator[idx_day], _preds_indicator[idx_day], average="macro"
    )
    f1_ind_night = f1_score(
        _targets_indicator[idx_night], _preds_indicator[idx_night], average="macro"
    )
    f1_ind_dusk = f1_score(
        _targets_indicator[idx_dawn_dusk],
        _preds_indicator[idx_dawn_dusk],
        average="macro",
    )
    f1_rear_day = f1_score(
        _targets_rear[idx_day], _preds_rear[idx_day], average="macro"
    )
    f1_rear_night = f1_score(
        _targets_rear[idx_night], _preds_rear[idx_night], average="macro"
    )
    f1_rear_dusk = f1_score(
        _targets_rear[idx_dawn_dusk], _preds_rear[idx_dawn_dusk], average="macro"
    )
    f1_heading_day = f1_score(
        _targets_heading[idx_day], _preds_heading[idx_day], average="macro"
    )
    f1_heading_night = f1_score(
        _targets_heading[idx_night], _preds_heading[idx_night], average="macro"
    )
    f1_heading_dusk = f1_score(
        _targets_heading[idx_dawn_dusk], _preds_heading[idx_dawn_dusk], average="macro"
    )
    print("Indicator")
    print("Day, Night, Dusk/Dawn")
    print(f"{f1_ind_day} & {f1_ind_night} & {f1_ind_dusk} \\")
    print("Rear")
    print("Day, Night, Dusk/Dawn")
    print(f"{f1_rear_day} & {f1_rear_night} & {f1_rear_dusk} \\")
    print("Heading")
    print("Day, Night, Dusk/Dawn")
    print(f"{f1_heading_day} & {f1_heading_night} & {f1_heading_dusk} \\")

    #
    # Weather
    #
    # Note: Not Implemented
    # Unfortunaetly the Waymo Open Dataset val test split does only contains sunny weather
