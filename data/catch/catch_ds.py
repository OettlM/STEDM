import cv2
import sys
import h5py
import torch
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from PIL import Image
from openslide import OpenSlide
from scipy.ndimage import binary_dilation, binary_erosion
from skimage.transform import downscale_local_mean

from data.catch.catch_utils import sample, wsi_sample



class CATCH_DS_Anno(torch.utils.data.Dataset):
    def __init__(self, img_l, anno_file_path, patch_size, base_scale, overlap, samples, lookup_f, num_classes, style_sampler, style_drop_rate, transforms=None):
        self._img_l = img_l
        self._anno_file_path = anno_file_path
        self._samples = samples
        self._patch_size = patch_size
        self._base_scale = base_scale
        self._lookup_f = lookup_f
        self._num_classes = num_classes
        self._style_sampler = style_sampler
        self._style_drop_rate = style_drop_rate
        self._transforms = transforms

        self.slide_objs = {}
        self.anno_file = None

        # get unique wsi nums
        wsi_nums = np.unique(np.array([el[3] for el in img_l]))
        wsi_num_dict = {}
        for i, wsi_num in enumerate(wsi_nums):
            wsi_num_dict[wsi_num] = i

        # create global sampling coordinates
        self._global_sample_list = [[] for i in range(self._num_classes+1)]
        for i in range(len(self._global_sample_list)):
            for j in range(len(wsi_nums)):
                self._global_sample_list[i].append([])

        p_h = int(((self._patch_size/2)*self._base_scale)/self._lookup_f)

        # create global sampling list
        for img_num, img_tup in enumerate(self._img_l):
            sampling_map = cv2.imdecode(img_tup[2], cv2.IMREAD_ANYDEPTH)

            for i in range(self._num_classes):
                mask = sampling_map[p_h:-p_h, p_h:-p_h]==i

                coords = np.argwhere(mask)
                coords += p_h
                coords = np.concatenate( (coords, np.full( (len(coords),1), img_num, dtype=coords.dtype)), axis=1)
                self._global_sample_list[i][wsi_num_dict[img_tup[3]]].append(coords)

            # for real white background
            mask = sampling_map[p_h:-p_h, p_h:-p_h]==255

            coords = np.argwhere(mask)
            coords += p_h
            coords = np.concatenate( (coords, np.full( (len(coords),1), img_num, dtype=coords.dtype)), axis=1)
            self._global_sample_list[-1][wsi_num_dict[img_tup[3]]].append(coords)

        # accumulate wsi_based sampling positions
        for i in range(len(self._global_sample_list)):
            for j in range(len(wsi_nums)):
                self._global_sample_list[i][j] = np.concatenate(self._global_sample_list[i][j], axis=0)

        # set correct sampling
        coords = []
        coords.append(np.concatenate(self._global_sample_list[0], axis=0))
        coords.append(np.concatenate([item for sublist in self._global_sample_list[1:-1] for item in sublist], axis=0))
        coords.append(np.concatenate(self._global_sample_list[-1], axis=0))

        self._global_sample_list = coords
        self._probs = np.array([0.4, 0.5, 0.1])

        # normalize sampling probs
        self._probs /= np.sum(self._probs)

        # create sampling list for style sampling
        self.style_samp_list = []
        # create list of unique file names
        catch_files = np.unique([el[0] for el in img_l])[::-1]

        self.kernel_size = (patch_size - 2*overlap) * self._base_scale

        # create coords
        for idx, file_name in enumerate(catch_files):
            slide = OpenSlide(str(file_name))

            ref_img = slide.read_region(location=(0,0), level=2, size=slide.level_dimensions[2])
            ref_img = np.min(np.array(ref_img)[:,:,:3], axis=2)
            sample_mask = downscale_local_mean(ref_img, (int(self.kernel_size / 16), int(self.kernel_size / 16)), cval=255)<230
            sample_mask = binary_dilation(sample_mask, iterations=2)
            sample_mask = binary_erosion(sample_mask, iterations=2)

            coords = np.argwhere(sample_mask)
            coords = np.concatenate((coords, np.full( (len(coords),1), idx, dtype=coords.dtype)), axis=1)

            self.style_samp_list.append(coords)

            slide.close()


    def __len__(self):
        if len(self._img_l) > 0:
            return self._samples
        else:
            return 0


    def __getitem__(self, idx):
        # sample class
        sampled_class = np.random.choice(len(self._probs), p=self._probs)
        class_list = self._global_sample_list[sampled_class]
        # sample instance
        sampled_instance = np.random.randint(0, len(class_list))
        patch_coords = class_list[sampled_instance]

        # get sampled image
        img_num = patch_coords[2]
        img_p = self._img_l[img_num]

        # create openslide object
        if self.slide_objs.get(img_p[-1]) is None:
            self.slide_objs[img_p[-1]] = OpenSlide(img_p[0])

        slide_obj = self.slide_objs[img_p[-1]]

        # open annotation h5py file
        if self.anno_file is None:
            self.anno_file = h5py.File(self._anno_file_path, mode="r")

        anno_dset = self.anno_file[str(img_p[-1])]

        # calculate sample coordinates
        y_c = int(patch_coords[0] * self._lookup_f + 0.5*self._lookup_f)
        x_c = int(patch_coords[1] * self._lookup_f + 0.5*self._lookup_f)

        # sample seg and patches
        sampled = sample(slide_obj, anno_dset, img_p[1], self._patch_size, self._base_scale, (y_c, x_c), self._transforms)
        img = sampled[0]
        seg = sampled[1]

        # sample style input
        style_list = self.style_samp_list[img_num]

        style_imgs = self._style_sampler.sample_imgs(slide_obj, (y_c, x_c), img_p[1], self._patch_size, self._base_scale, style_list, self.kernel_size)

        one_hot = F.one_hot(seg.to(torch.long), num_classes=self._num_classes).permute(2,0,1).to(torch.float32)

        # randomly drop style
        if np.random.uniform(0, 1.0) < self._style_drop_rate:
            style_imgs = torch.zeros_like(style_imgs)-0.5

        return img*2-1, one_hot, seg, style_imgs*2-1


class CATCH_DS_Predict(CATCH_DS_Anno):
    def __getitem__(self, idx):
        return *super().__getitem__(idx), idx


class CATCH_DS_Ordered(torch.utils.data.Dataset):
    def __init__(self, img_l, anno_file_path, patch_size, base_scale, overlap, lookup_f, wsi_red_factor, num_classes, transforms=None):
        self._img_l = img_l
        self._anno_file_path = anno_file_path
        self._patch_size = patch_size
        self._base_scale = base_scale
        self._overlap = overlap
        self._lookup_f = lookup_f
        self._wsi_red_factor = wsi_red_factor
        self._num_classes = num_classes
        self._transforms = transforms

        self.kernel_size = (patch_size - 2*overlap) * self._base_scale

        self.slide_obj = None
        self.curr_slide = None
        self.anno_file = None
        
        self.slide_info = []

        self._sample_list = np.zeros((0,3), dtype=np.int64)
        patch_f = int((self.kernel_size)/lookup_f)
        for img_num, img_tup in enumerate(self._img_l):
            sampling_map = cv2.imdecode(img_tup[2], cv2.IMREAD_ANYDEPTH)
            sampling_mask = np.ones(sampling_map.shape, dtype=np.uint8)
            border = int((overlap*self._base_scale) / lookup_f)
            sampling_mask = sampling_mask[border:-border, border:-border]
            block_y = int(sampling_mask.shape[0] / patch_f)
            block_x = int(sampling_mask.shape[1] / patch_f)
            sampling_mask = sampling_mask[:block_y, :block_x]

            # reduce number of sampling points
            sampling_mask = sampling_mask[::self._wsi_red_factor, ::self._wsi_red_factor]

            # create sampling positions
            coords = np.argwhere(sampling_mask)
            coords *= self._wsi_red_factor
            coords *= patch_f
            coords += border
            coords = np.concatenate( (coords, np.full( (len(coords),1), img_num, dtype=coords.dtype)), axis=1)
            self._sample_list = np.concatenate((self._sample_list, coords), axis=0)

        unique_wsi = np.unique(np.array([img_tup[-1] for img_tup in self._img_l]))
        self.wsi_lookup = {}
        for num, wsi_num in enumerate(unique_wsi):
            self.wsi_lookup[wsi_num] = num


    def __len__(self):
        if len(self._img_l) > 0:
            return len(self._sample_list)
        else:
            return 0

    
    def __getitem__(self, idx):
        patch_coords = self._sample_list[idx]

        # get sampled image
        img_num = patch_coords[2]
        img_p = self._img_l[img_num]

        # create openslide object
        if self.curr_slide is None:
            self.slide_obj = OpenSlide(img_p[0])
            self.curr_slide = img_p[0]
        elif self.curr_slide != img_p[0]:
            self.slide_obj.close()
            self.slide_obj = OpenSlide(img_p[0])
            self.curr_slide = img_p[0]

        slide_obj = self.slide_obj

        # open annotation h5py file
        if self.anno_file is None:
            self.anno_file = h5py.File(self._anno_file_path, mode="r")

        anno_dset = self.anno_file[str(img_p[-1])]

        # calculate sample coordinates
        y_c = int(patch_coords[0] * self._lookup_f + 0.5*self.kernel_size)
        x_c = int(patch_coords[1] * self._lookup_f + 0.5*self.kernel_size)

        # sample seg and patches
        sampled = sample(slide_obj, anno_dset, img_p[1], self._patch_size, self._base_scale, (y_c, x_c), self._transforms)
        img = sampled[0]
        seg = sampled[1]
        one_hot = F.one_hot(seg.to(torch.long), num_classes=self._num_classes).permute(2,0,1).to(torch.float32)

        return img*2-1, one_hot, seg, (0, 0), self.wsi_lookup[img_p[-1]]


class CATCH_DS_UnAnno(torch.utils.data.Dataset):
    def __init__(self, img_l, patch_size, base_scale, overlap, samples, num_classes, style_sampler, transforms=None):
        self._img_l = img_l
        self._samples = samples
        self._patch_size = patch_size
        self._base_scale = base_scale
        self._overlap = overlap
        self._num_classes = num_classes
        self._style_sampler = style_sampler
        self._transforms = transforms

        self.slide_objs = {}

        self.kernel_size = (patch_size - 2*overlap) * self._base_scale

        self._sample_list = []

        # create list for all patch numbers, OpenSlide objects, scores
        print("Setting up CATCH unannotated dataset")
        sys.stdout.flush()
        for idx, img_tup in tqdm(enumerate(self._img_l), desc="Processing CATCH WSI", total=len(self._img_l)):
            slide = OpenSlide(img_tup[0])

            level = 2
            ref_img = slide.read_region(location=(0,0), level=level, size=slide.level_dimensions[level])
            ref_img = np.min(np.array(ref_img)[:,:,:3], axis=2)
            sample_mask = downscale_local_mean(ref_img, (int(self.kernel_size / slide.level_downsamples[level]), int(self.kernel_size / slide.level_downsamples[level])), cval=255)<230
            sample_mask = binary_dilation(sample_mask, iterations=2)
            sample_mask = binary_erosion(sample_mask, iterations=2)

            coords = np.argwhere(sample_mask)
            coords = np.concatenate((coords, np.full( (len(coords),1), idx, dtype=coords.dtype)), axis=1)

            self._sample_list.append(coords)

            slide.close()


    def __len__(self):
        if len(self._sample_list) > 0:
            return self._samples
        else:
            return 0


    def __getitem__(self, idx):
        # sample instance (WSI)
        sampled_instance = np.random.randint(0, len(self._sample_list))
        instance_list = self._sample_list[sampled_instance]
        # sample position
        sampled_coords = np.random.randint(0, len(instance_list))
        patch_coords = instance_list[sampled_coords]

        # get sampled image
        img_num = patch_coords[2]
        img_p = self._img_l[img_num]

        # create openslide object
        if self.slide_objs.get(img_p[-1]) is None:
            self.slide_objs[img_p[-1]] = OpenSlide(img_p[0])

        slide_obj = self.slide_objs[img_p[-1]]

        y_c = (patch_coords[0] + 0.5) * self.kernel_size
        x_c = (patch_coords[1] + 0.5) * self.kernel_size
        
        img_crops = wsi_sample(slide_obj, (0,0), self._patch_size, self._base_scale, (y_c, x_c))

        applied = self._transforms(image=img_crops)
        img = applied['image']
        seg = torch.full((self._patch_size, self._patch_size), -1, dtype=torch.long)

        # sample style input
        style_imgs = self._style_sampler.sample_imgs(slide_obj, (y_c, x_c), img_p[1], self._patch_size, self._base_scale, instance_list, self.kernel_size)

        one_hot = torch.zeros((self._num_classes, self._patch_size, self._patch_size), dtype=torch.float32)

        return img*2-1, one_hot, seg, style_imgs*2-1


class CATCH_DS_Syn(torch.utils.data.Dataset):
    def __init__(self, img_l, samples, num_classes, transforms=None):
        self._img_l = img_l
        self._samples = samples
        self._num_classes = num_classes
        self._transforms = transforms


    def __len__(self):
        if len(self._img_l) > 0:
            return self._samples
        else:
            return 0


    def __getitem__(self, idx):
        sample_idx = np.random.randint(0, len(self._img_l))
        img_tup = self._img_l[sample_idx]

        img = np.array(Image.open(img_tup[0]))
        seg = np.array(Image.open(img_tup[1]).convert('L'))

        applied = self._transforms(image=img, mask=seg)
        img = applied['image']
        seg = applied['mask']
        one_hot = F.one_hot(seg.to(torch.long), num_classes=self._num_classes).permute(2,0,1).to(torch.float32)
        style = torch.zeros((1,3,512,512), dtype=torch.float32)

        return img*2-1, one_hot, seg, style