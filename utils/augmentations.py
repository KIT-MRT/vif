import math
import random

import torch
from torch.utils.data._utils.collate import default_collate
import torchvision.transforms.v2 as transforms
import torchvision
torchvision.disable_beta_transforms_warning()


def flip_angle_radians(angle_rad):
    flipped_angle = -angle_rad
    flipped_angle[flipped_angle < -math.pi] = (
        flipped_angle[flipped_angle < -math.pi] + 2 * math.pi
    )
    return flipped_angle


def augment_sequences(batch):
    flip = RandomHorizontalFlip()
    t = transforms.RandomPhotometricDistort()
    # t = transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)
    eraze = RandomErazing()
    blur = transforms.GaussianBlur(kernel_size=5)

    batch_flipped = []
    for images, headings, target in batch:
        images, headings, target = flip(images, headings, target)
        images, headings, target = eraze(images, headings, target)

        if random.randint(0, 9) < 5:
            images = t(images)

        if random.randint(0, 9) < 5:
            images = blur(images)

        batch_flipped.append([images, headings, target])

    return default_collate(batch_flipped)


class RandomHorizontalFlip:
    def __init__(self):
        self.flip = torchvision.transforms.RandomHorizontalFlip(p=1.0)

    def __call__(self, images, headings, target):
        if random.randint(0, 9) < 5:
            return images, headings, target

        images_flipped = []
        for image in images:
            images_flipped.append(self.flip(image))

        indicator_l = target[1].item()
        indicator_r = target[2].item()
        if indicator_l == 1:
            target[1] = 0
            target[2] = 1

        if indicator_r == 1:
            target[2] = 0
            target[1] = 1

        heading_l = target[8].item()
        heading_r = target[9].item()
        if heading_l == 1:
            target[8] = 0
            target[9] = 1

        if heading_r == 1:
            target[9] = 0
            target[8] = 1

        headings = flip_angle_radians(headings)

        return torch.stack(images_flipped), headings, target


class RandomErazing:
    def __init__(self):
        self.eraze = transforms.RandomErasing(
            p=1.0, scale=(0.02, 0.2), ratio=(0.3, 3.3)
        )

    def __call__(self, images, headings, target):
        if random.randint(0, 9) < 4:
            return images, headings, target

        return self.eraze(images), headings, target
