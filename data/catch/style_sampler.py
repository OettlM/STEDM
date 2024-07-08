import torch
import numpy as np

from data.catch.catch_utils import wsi_sample


class NoneSampler:
    def __init__(self, cfg, transforms) -> None:
        self._cfg = cfg
        self._transforms = transforms

    def sample_imgs(self, slide_obj, pos, offset, p_size, b_scale, sample_list, lookup_f):
        return self._transforms(image=np.zeros((p_size, p_size, 3), dtype=np.uint8))["image"].unsqueeze(0)


class NearbySampler:
    def __init__(self, cfg, transforms) -> None:
        self._cfg = cfg
        self._transforms = transforms

        self._relative_dist = cfg.relative_dist

    def sample_imgs(self, slide_obj, pos, offset, p_size, b_scale, sample_list, lookup_f):
        y_c, x_c = pos

        y_c += self._relative_dist*np.random.randint(-p_size*b_scale, p_size*b_scale)
        x_c += self._relative_dist*np.random.randint(-p_size*b_scale, p_size*b_scale)

        style_crop = wsi_sample(slide_obj, offset, p_size, b_scale, (y_c, x_c))

        style_crop = self._transforms(image=style_crop)["image"][None,:]
        return style_crop


class MultiPatchSampler:
    def __init__(self, cfg, transforms) -> None:
        self._cfg = cfg
        self._transforms = transforms

        self._num_patches = cfg.num_patches

    def sample_imgs(self, slide_obj, pos, offset, p_size, b_scale, sample_list, lookup_f):
        style_imgs = []
        for i in range(self._num_patches):
            sampled_coords = np.random.randint(0, len(sample_list))
            patch_coords = sample_list[sampled_coords]

            # calculate sample coordinates
            y_c = int(patch_coords[0] * lookup_f + 0.5*lookup_f) + np.random.randint(-lookup_f, lookup_f)
            x_c = int(patch_coords[1] * lookup_f + 0.5*lookup_f) + np.random.randint(-lookup_f, lookup_f)

            # sample seg and patch
            style_crop = wsi_sample(slide_obj, (0,0), p_size, b_scale, (y_c, x_c))

            # apply basic augments
            style_crop = self._transforms(image=style_crop)["image"]
            style_imgs.append(style_crop)

        style_imgs = torch.stack(style_imgs, axis=0)
        return style_imgs