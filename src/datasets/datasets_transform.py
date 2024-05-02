import random

import torch.nn as nn
import torchvision.transforms as transforms

import open_clip


class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


def _convert_to_rgb(image):
    return image.convert("RGB")


def openclip_transform(image_size=448, is_train=False):
    return open_clip.image_transform(
        image_size,
        is_train=is_train,
    )


def wilds_default_train_transform(image_size=448):
    return transforms.Compose(
        [
            transforms.Resize(
                image_size, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(),
            RandomApply(transforms.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.3),
            transforms.CenterCrop(image_size),
            _convert_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


def fmow_train_transform(image_size=448):

    return transforms.Compose(
        [
            transforms.Resize(
                image_size, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.CenterCrop(image_size),
            _convert_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


def fmow_test_transform(image_size=448):

    return transforms.Compose(
        [
            transforms.Resize(
                image_size, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(image_size),
            _convert_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


def domainnet_train_transform(image_size=448):

    if image_size == 224:
        resize = 256
    elif image_size == 336:
        resize = 384

    return transforms.Compose(
        [
            transforms.Resize(
                resize, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


def domainnet_test_transform(image_size=448):

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            _convert_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


def camelyon_train_transform(image_size=448):

    return transforms.Compose(
        [
            transforms.Resize(
                image_size, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.CenterCrop(image_size),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
            RandomApply(transforms.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.3),
        ]
    )
