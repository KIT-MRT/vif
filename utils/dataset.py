import pickle
import os.path

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from skimage import io
import numpy as np
from imagesize import imagesize


def create_data_dict(
    annotations_pkl,
    images_dir,
    sequence_len=20,
    only_intentions=False,
    keep_none=True,
    max_occupancy=None,
    min_image_width=None,
    min_image_height=None,
    cache_file=False,
    ignore_extraction_issues=True,
):
    #
    # Return cached sequences
    #
    if cache_file and os.path.isfile(cache_file):
        with open(cache_file, "rb") as fh:
            data_dict = pickle.load(fh)

        return data_dict

    with open(annotations_pkl, "rb") as f:
        annotations = pickle.load(f)

    #
    # Filter sequences which are shorter than $sequence_len
    #
    short_keys = []
    for _, scene_id in enumerate(annotations):
        for _, object_id in enumerate(annotations[scene_id]):
            if object_id == "stats":
                continue

            if len(annotations[scene_id][object_id]) < sequence_len:
                short_keys.append((scene_id, object_id))

    for k in short_keys:
        del annotations[k[0]][k[1]]

    #
    # Create combinations of all possible sequences
    #
    data_dict = {}
    data_dict["len"] = 0
    data_targets = []
    for _, scene_id in tqdm(enumerate(annotations)):
        weather = annotations[scene_id]["stats"]["weather"]
        time_of_day = annotations[scene_id]["stats"]["time_of_day"]

        for _, object_id in enumerate(annotations[scene_id]):
            if object_id == "stats":
                continue

            labels = []
            image_files = []
            images_width = []
            images_height = []
            some_indication = False

            # check if object dir excists on disk
            if not os.path.exists(f"{images_dir}{scene_id}/{object_id}"):
                if not ignore_extraction_issues:
                    raise RuntimeError(
                        f"Object: {images_dir}{scene_id}/{object_id} is annotated but does not exists on disk"
                    )
                continue
            #
            # extract information of regarding all frames for object_id
            #
            for timestep in annotations[scene_id][object_id]:
                image_file = f"{images_dir}{scene_id}/{object_id}/{timestep}.jpeg"
                left = annotations[scene_id][object_id][timestep]["left"]
                right = annotations[scene_id][object_id][timestep]["right"]
                emergency = annotations[scene_id][object_id][timestep]["emergency"]
                breaking = annotations[scene_id][object_id][timestep]["break"]
                rear = annotations[scene_id][object_id][timestep]["rear"]
                heading = annotations[scene_id][object_id][timestep]["heading"]
                occ = annotations[scene_id][object_id][timestep]["occ"]

                # image stats
                width, height = imagesize.get(image_file)
                image_files.append(image_file)
                images_width.append(width)
                images_height.append(height)

                none_indicator = (
                    0 if int(left) + int(right) + int(emergency) >= 1 else 1
                )
                none_rear = 0 if int(breaking) + int(rear) >= 1 else 1

                if none_indicator == 0:
                    some_indication = True

                heading = float(heading)
                if -0.785 < heading < 0.785:
                    heading_vector = [1, 0, 0, 0]
                elif 0.785 <= heading < 2.355:
                    heading_vector = [0, 1, 0, 0]
                elif -2.355 <= heading < -0.785:
                    heading_vector = [0, 0, 1, 0]
                else:
                    heading_vector = [0, 0, 0, 1]

                # vehicle annotation
                labels.append(
                    [
                        none_indicator,
                        int(left),
                        int(right),
                        int(emergency),
                        none_rear,
                        int(breaking),
                        int(rear),
                        heading_vector[0],
                        heading_vector[1],
                        heading_vector[2],
                        heading_vector[3],
                        occ,
                    ]
                )

            #
            # Extract Sequences of object_id
            #
            if only_intentions and not some_indication:
                continue

            sequences = []
            targets = []
            for start_idx in range(len(labels)):
                end_idx = start_idx + sequence_len - 1
                if end_idx >= len(labels):
                    break

                # Skip all sequences that do not end with a indicator intention
                if not keep_none:
                    target = labels[end_idx]
                    if target[0] == 1:
                        continue

                #
                # Check image width and occupancy
                #
                ok = True
                sequence_images = []
                sequence_headings = np.array(labels)[start_idx : end_idx + 1, 7]
                target = labels[end_idx]
                for idx in range(start_idx, end_idx + 1):
                    # image width
                    if (
                        min_image_width is not None
                        and images_width[idx] < min_image_width
                    ):
                        ok = False
                        break

                    if (
                        min_image_height is not None
                        and images_height[idx] < min_image_height
                    ):
                        ok = False
                        break

                    # occupancy
                    if max_occupancy and labels[idx][11] > max_occupancy:
                        ok = False
                        break

                    sequence_images.append(image_files[idx])

                if ok:
                    sequences.append([sequence_images, sequence_headings])
                    targets.append(target)

            for idx in range(len(targets)):
                dict_idx = data_dict["len"] + idx
                data_dict[dict_idx] = {}
                data_dict[dict_idx]["sequence_images"] = sequences[idx][0]
                data_dict[dict_idx]["sequence_headings"] = sequences[idx][1]
                data_dict[dict_idx]["target"] = targets[idx]
                data_targets.append(targets[idx])
                data_dict[dict_idx]["weather"] = weather
                data_dict[dict_idx]["time_of_day"] = time_of_day
            data_dict["len"] += len(targets)

    none = sum(np.array(data_targets)[:, 0])
    left = sum(np.array(data_targets)[:, 1])
    right = sum(np.array(data_targets)[:, 2])
    emergency = sum(np.array(data_targets)[:, 3])

    print(
        f"dataset stats: none {none}, left {left}, right {right}, emergency: {emergency}"
    )

    data_dict["indicator_class_none"] = 1 / (none + int(only_intentions))
    data_dict["indicator_class_left"] = 1 / left
    data_dict["indicator_class_right"] = 1 / right
    data_dict["indicator_class_emergency"] = 1 / emergency

    if cache_file:
        with open(cache_file, "wb") as fh:
            pickle.dump(data_dict, fh, protocol=pickle.HIGHEST_PROTOCOL)

    return data_dict


class VisualIntentionsDictDataset(Dataset):
    def __init__(
        self, data_dict, image_transform=None, full_transform=None, with_context=False
    ):
        self.data_dict = data_dict
        self.image_transform = image_transform
        self.full_transform = full_transform
        self.with_context = with_context

    def __len__(self):
        return self.data_dict["len"]

    def __getitem__(self, idx):
        sample = self.data_dict[idx]

        images = []
        for image in sample["sequence_images"]:
            image = io.imread(image)
            if self.image_transform:
                image = self.image_transform(image)
            images.append(image)

        headings = sample["sequence_headings"]

        if self.with_context:
            return (
                torch.stack(images),
                torch.tensor(headings, dtype=torch.float32),
                torch.Tensor(sample["target"]),
                sample["weather"],
                sample["time_of_day"],
            )
        return (
            torch.stack(images),
            torch.tensor(headings, dtype=torch.float32),
            torch.Tensor(sample["target"]),
        )
