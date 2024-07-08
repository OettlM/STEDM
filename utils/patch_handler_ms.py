import math

import numpy as np



class PatchHandlerMS:
    def __init__(self, img_shape, patch_size, overlap):
        self._img_shape = img_shape
        self._patch_size = patch_size
        self._kernel_size = patch_size - 2*overlap
        self._overlap = overlap

        self._x_p = math.ceil(img_shape[0] / self._kernel_size)
        self._y_p = math.ceil(img_shape[1] / self._kernel_size)
        self._img_p_num = int(self._x_p * self._y_p)

        self._out_image = None
        self._created = False
    
    def num_segs(self):
        return self._img_p_num

    def get(self, image, idx, scale):
        j = int(idx / self._x_p)
        i = int(idx % self._x_p)

        adj_kernel = self._kernel_size / scale
        adj_overlap = self._overlap / scale
        x_s = int(i * adj_kernel - adj_overlap)
        y_s = int(j * adj_kernel - adj_overlap)
        x_e = int(x_s + self._patch_size)
        y_e = int(y_s + self._patch_size)

        x_s_mod = int(max(x_s, 0) - x_s)
        y_s_mod = int(max(y_s, 0) - y_s)
        x_e_mod = int(x_e - min(x_e, image.shape[1]))
        y_e_mod = int(y_e - min(y_e, image.shape[0]))

        # we use ones as padding, since microsopy images are (1,1,1) in empty regions
        if len(image.shape) > 2:
            out_patch = np.ones((self._patch_size, self._patch_size, image.shape[2]), dtype=image.dtype)*255
        else:
            out_patch = np.zeros((self._patch_size, self._patch_size), dtype=image.dtype)
        
        img_patch = image[y_s+y_s_mod : y_e-y_e_mod, x_s+x_s_mod : x_e-x_e_mod]
        out_patch[y_s_mod:self._patch_size-y_e_mod, x_s_mod:self._patch_size-x_e_mod] = img_patch
        return out_patch

    def take(self, patch, idx):
        if not self._created:
            self._created = True
            if len(patch.shape) > 2:
                self._out_image = np.zeros((self._img_shape[0], self._img_shape[1], patch.shape[2]), dtype=patch.dtype)
            else:
                self._out_image = np.zeros((self._img_shape[0], self._img_shape[1]), dtype=patch.dtype)

        j = int(idx / self._x_p)
        i = int(idx % self._x_p)
        o = self._overlap

        x_s = int(i * self._kernel_size - self._overlap)
        y_s = int(j * self._kernel_size - self._overlap)
        x_e = int(x_s + self._kernel_size + 2 * self._overlap)
        y_e = int(y_s + self._kernel_size + 2 * self._overlap)

        x_s_mod = int(max(x_s, 0) - x_s)
        y_s_mod = int(max(y_s, 0) - y_s)
        x_e_mod = int(x_e - min(x_e, self._img_shape[1]))
        y_e_mod = int(y_e - min(y_e, self._img_shape[0]))

        self._out_image[y_s+o : min(self._out_image.shape[0],y_e-o), x_s+o : min(self._out_image.shape[1],x_e-o)] = patch[o:min(self._patch_size-o, self._patch_size-y_e_mod), o:min(self._patch_size-o, self._patch_size-x_e_mod)]

    def get_out_image(self):
        return self._out_image

    def clear(self):
        self._out_image = None
        self._created = False