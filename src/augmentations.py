# -*- coding: utf-8 -*-
"""
Augmentation strategies for semi-supervised learning on CIFAR-10.

Provides:
  - Weak augmentation  : random horizontal flip + random crop (used in both papers)
  - Strong augmentation: RandAugment + Cutout (used in FixMatch)
  - Additional strategies for B/A grade analysis:
      color_jitter, gaussian_blur, random_rotation, grayscale, solarize

References:
  - FixMatch (Sohn et al., 2020) — Appendix E, Tables 12-13
  - RandAugment (Cubuk et al., 2019)
  - Cutout (DeVries & Taylor, 2017)
"""

import random
import numpy as np

import torch
import torchvision.transforms as T
from PIL import Image, ImageEnhance, ImageOps


# ---------------------------------------------------------------------------
# CIFAR-10 normalisation constants
# ---------------------------------------------------------------------------

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2471, 0.2435, 0.2616)

_normalise = T.Normalize(CIFAR10_MEAN, CIFAR10_STD)


# ---------------------------------------------------------------------------
# Paper-specified augmentations
# ---------------------------------------------------------------------------

def get_weak_augmentation():
    """
    Weak augmentation as described in both MixMatch and FixMatch papers:
      random horizontal flip (p=0.5) + random crop (±4 pixels / 12.5%)
    """
    return T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(32, padding=4, padding_mode="reflect"),
        T.ToTensor(),
        _normalise,
    ])


def get_strong_augmentation():
    """
    Strong augmentation as used in FixMatch (RandAugment version):
      random flip + crop → RandAugment(N=2, M=10) → Cutout
    """
    return T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(32, padding=4, padding_mode="reflect"),
        RandAugment(n=2, m=10),
        T.ToTensor(),
        _normalise,
        Cutout(n_holes=1, length=16),
    ])


def get_test_transform():
    """Deterministic transform for evaluation (normalise only)."""
    return T.Compose([
        T.ToTensor(),
        _normalise,
    ])


# ---------------------------------------------------------------------------
# Additional augmentations for B/A analysis
# ---------------------------------------------------------------------------

def get_color_jitter_augmentation(strength: float = 0.5):
    """
    Weak aug + ColorJitter.
    Tests whether color distortions improve semi-supervised performance.
    """
    return T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(32, padding=4, padding_mode="reflect"),
        T.ColorJitter(
            brightness=strength,
            contrast=strength,
            saturation=strength,
            hue=strength * 0.5,
        ),
        T.ToTensor(),
        _normalise,
    ])


def get_gaussian_blur_augmentation(sigma: tuple = (0.1, 2.0)):
    """
    Weak aug + Gaussian blur.
    Forces the model to use structural rather than texture features.
    """
    return T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(32, padding=4, padding_mode="reflect"),
        T.GaussianBlur(kernel_size=3, sigma=sigma),
        T.ToTensor(),
        _normalise,
    ])


def get_rotation_augmentation(degrees: float = 15.0):
    """
    Weak aug + random rotation.
    Tests rotation invariance as an additional inductive bias.
    """
    return T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(32, padding=4, padding_mode="reflect"),
        T.RandomRotation(degrees=degrees),
        T.ToTensor(),
        _normalise,
    ])


def get_grayscale_augmentation(p: float = 0.2):
    """
    Weak aug + random grayscale conversion.
    Encourages learning of shape/texture features over colour.
    """
    return T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(32, padding=4, padding_mode="reflect"),
        T.RandomGrayscale(p=p),
        T.ToTensor(),
        _normalise,
    ])


def get_combined_strong_augmentation():
    """
    Weak aug + RandAugment + Cutout + ColorJitter.
    A maximally aggressive strategy for the B/A augmentation study.
    """
    return T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(32, padding=4, padding_mode="reflect"),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        RandAugment(n=2, m=10),
        T.ToTensor(),
        _normalise,
        Cutout(n_holes=1, length=16),
    ])


# Registry: name → callable that returns a transform
AUGMENTATION_REGISTRY = {
    "weak":         get_weak_augmentation,
    "strong":       get_strong_augmentation,
    "color_jitter": get_color_jitter_augmentation,
    "gaussian_blur": get_gaussian_blur_augmentation,
    "rotation":     get_rotation_augmentation,
    "grayscale":    get_grayscale_augmentation,
    "combined":     get_combined_strong_augmentation,
}


def get_augmentation(name: str):
    """Return an augmentation transform by name (see AUGMENTATION_REGISTRY)."""
    if name not in AUGMENTATION_REGISTRY:
        raise ValueError(f"Unknown augmentation '{name}'. "
                         f"Choose from: {list(AUGMENTATION_REGISTRY.keys())}")
    return AUGMENTATION_REGISTRY[name]()


# ---------------------------------------------------------------------------
# RandAugment
# ---------------------------------------------------------------------------

class RandAugment:
    """
    RandAugment: randomly applies N transforms, each with magnitude M.

    As used in FixMatch (Sohn et al. 2020), we sample the magnitude randomly
    from [0, M] at each step rather than using a fixed global value (following
    the UDA / RandAugment-with-random-magnitude variant described in Appendix E
    of the FixMatch paper).

    Transformations match Table 12 of the FixMatch paper.
    """

    def __init__(self, n: int = 2, m: int = 10):
        assert 0 < n
        assert 0 <= m <= 30
        self.n = n
        self.m = m
        self._ops = self._build_ops()

    # ------------------------------------------------------------------
    # Individual augmentation ops (PIL → PIL)
    # ------------------------------------------------------------------

    @staticmethod
    def _autocontrast(img, _v):
        return ImageOps.autocontrast(img)

    @staticmethod
    def _brightness(img, v):
        return ImageEnhance.Brightness(img).enhance(
            max(0.05, 1.0 + v * random.choice([-1, 1]) * 0.1)
        )

    @staticmethod
    def _color(img, v):
        return ImageEnhance.Color(img).enhance(
            max(0.05, 1.0 + v * random.choice([-1, 1]) * 0.1)
        )

    @staticmethod
    def _contrast(img, v):
        return ImageEnhance.Contrast(img).enhance(
            max(0.05, 1.0 + v * random.choice([-1, 1]) * 0.1)
        )

    @staticmethod
    def _equalize(img, _v):
        return ImageOps.equalize(img)

    @staticmethod
    def _identity(img, _v):
        return img

    @staticmethod
    def _posterize(img, v):
        bits = max(1, int(8 - v * 0.4))
        return ImageOps.posterize(img, bits)

    @staticmethod
    def _rotate(img, v):
        angle = v * 3.0 * random.choice([-1, 1])
        return img.rotate(angle, resample=Image.BILINEAR)

    @staticmethod
    def _sharpness(img, v):
        return ImageEnhance.Sharpness(img).enhance(
            max(0.05, 1.0 + v * random.choice([-1, 1]) * 0.1)
        )

    @staticmethod
    def _shear_x(img, v):
        shear = v * 0.03 * random.choice([-1, 1])
        return img.transform(img.size, Image.AFFINE,
                             (1, shear, 0, 0, 1, 0),
                             resample=Image.BILINEAR)

    @staticmethod
    def _shear_y(img, v):
        shear = v * 0.03 * random.choice([-1, 1])
        return img.transform(img.size, Image.AFFINE,
                             (1, 0, 0, shear, 1, 0),
                             resample=Image.BILINEAR)

    @staticmethod
    def _solarize(img, v):
        threshold = int(256 - v * 25.6)
        return ImageOps.solarize(img, threshold)

    @staticmethod
    def _translate_x(img, v):
        dx = int(v * 0.03 * img.size[0] * random.choice([-1, 1]))
        return img.transform(img.size, Image.AFFINE,
                             (1, 0, dx, 0, 1, 0),
                             resample=Image.BILINEAR)

    @staticmethod
    def _translate_y(img, v):
        dy = int(v * 0.03 * img.size[1] * random.choice([-1, 1]))
        return img.transform(img.size, Image.AFFINE,
                             (1, 0, 0, 0, 1, dy),
                             resample=Image.BILINEAR)

    def _build_ops(self):
        return [
            self._autocontrast,
            self._brightness,
            self._color,
            self._contrast,
            self._equalize,
            self._identity,
            self._posterize,
            self._rotate,
            self._sharpness,
            self._shear_x,
            self._shear_y,
            self._solarize,
            self._translate_x,
            self._translate_y,
        ]

    def __call__(self, img: Image.Image) -> Image.Image:
        selected = random.choices(self._ops, k=self.n)
        for op in selected:
            # Sample magnitude uniformly in [1, M]
            v = random.uniform(1, self.m)
            img = op(img, v)
        return img


# ---------------------------------------------------------------------------
# Cutout
# ---------------------------------------------------------------------------

class Cutout:
    """
    Randomly masks a square patch of an image tensor.

    Reference: DeVries & Taylor, "Improved Regularization of Convolutional
    Neural Networks with Cutout", 2017.

    Applied *after* ToTensor() and Normalize().
    """

    def __init__(self, n_holes: int = 1, length: int = 16):
        self.n_holes = n_holes
        self.length  = length

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        # img: (C, H, W)
        _, h, w = img.shape
        mask = torch.ones_like(img)
        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = max(0, y - self.length // 2)
            y2 = min(h, y + self.length // 2)
            x1 = max(0, x - self.length // 2)
            x2 = min(w, x + self.length // 2)
            mask[:, y1:y2, x1:x2] = 0.0
        return img * mask
