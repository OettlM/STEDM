import numpy as np


class NoneSampler:
    def __init__(self, cfg, transforms) -> None:
        self._cfg = cfg
        self._transforms = transforms

    def sample_imgs(self, img):
        return self._transforms(image=np.zeros(img.shape, dtype=np.uint8))["image"].unsqueeze(0)


class AugmentedSampler:
    def __init__(self, cfg, transforms) -> None:
        self._cfg = cfg
        self._transforms = transforms

    def sample_imgs(self, img):
        return self._transforms(image=img)["image"].unsqueeze(0)