import cv2
import sys
import torch

import numpy as np
import pandas as pd
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm
from openslide import OpenSlide
from scipy.ndimage import binary_dilation, binary_erosion, binary_opening
from skimage.transform import downscale_local_mean

from utils.patch_handler_ms import PatchHandlerMS
from data.her2.her2_utils import sample, wsi_sample

        

class HER2_DS_Anno(torch.utils.data.Dataset):
    def __init__(self, img_l, samples, patch_size, base_scale, overlap, lookup_f, num_classes, style_sampler, style_drop_rate, transforms=None):
        self._img_l = img_l
        self._samples = samples
        self._patch_size = patch_size
        self._base_scale = base_scale
        self._lookup_f = lookup_f
        self._num_classes = num_classes
        self._style_sampler = style_sampler
        self._style_drop_rate = style_drop_rate
        self._transforms = transforms

        self.slide_objs = {}

        # get unique wsi nums
        wsi_nums = np.unique(np.array([el[4] for el in img_l]))
        wsi_num_dict = {}
        for i, wsi_num in enumerate(wsi_nums):
            wsi_num_dict[wsi_num] = i

        # create global sampling coordinates
        self._global_sample_list = [[] for i in range(self._num_classes)]
        for i in range(len(self._global_sample_list)):
            for j in range(len(wsi_nums)):
                self._global_sample_list[i].append([])

        p_h = int(((self._patch_size/2)*self._base_scale)/self._lookup_f)

        for img_num, img_tup in enumerate(self._img_l):
            # look lookup image
            sampling_map = cv2.imdecode(img_tup[3], cv2.IMREAD_ANYDEPTH)
            # collect sampling pos for each class
            for i in range(self._num_classes):
                mask = sampling_map[p_h:-p_h, p_h:-p_h]==i

                coords = np.argwhere(mask)
                coords += p_h
                coords = np.concatenate( (coords, np.full( (len(coords),1), img_num, dtype=coords.dtype)), axis=1)

                # save coords
                self._global_sample_list[i][wsi_num_dict[img_tup[4]]].append(coords)

        # accumulate wsi_based sampling positions
        for i in range(len(self._global_sample_list)):
            for j in range(len(wsi_nums)):
                self._global_sample_list[i][j] = np.concatenate(self._global_sample_list[i][j], axis=0)

        # set correct sampling
        coords = []
        coords.append(self._global_sample_list[0])

        tmp_list = []
        for i in range(len(self._global_sample_list[1])):
            samp_coords = self._global_sample_list[1][i]
            for j in range(2, len(self._global_sample_list)):
                samp_coords = np.concatenate((samp_coords, self._global_sample_list[j][i]), axis=0)
            
            tmp_list.append(samp_coords)

        coords.append(tmp_list)

        self._global_sample_list = coords
        self._probs = np.array([0.5, 0.5])

        # normalize sampling probs
        self._probs /= np.sum(self._probs)

        # create sampling list for style sampling
        self.style_samp_list = []
        # create list of unique file names
        her2_files = np.unique([el[0] for el in img_l])[::-1]

        self.kernel_size = (patch_size - 2*overlap) * self._base_scale

        # create coords
        for idx, file_name in enumerate(her2_files):
            slide = OpenSlide(str(file_name))

            ref_img = slide.read_region(location=(0,0), level=2, size=slide.level_dimensions[2])
            ref_img = np.min(np.array(ref_img)[:,:,:3], axis=2)
            sample_mask = downscale_local_mean(ref_img, (int(self.kernel_size / 16), int(self.kernel_size / 16)), cval=255)<253
            sample_mask = binary_dilation(sample_mask, iterations=2)
            sample_mask = binary_erosion(sample_mask, iterations=2)
            sample_mask = binary_opening(sample_mask, iterations=2)

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
        # sample instance (WSI)
        sampled_instance = np.random.randint(0, len(class_list))
        instance_list = class_list[sampled_instance]
        # sample position
        sampled_coords = np.random.randint(0, len(instance_list))
        patch_coords = instance_list[sampled_coords]

        # get sampled image
        img_num = patch_coords[2]
        img_p = self._img_l[img_num]

        if self.slide_objs.get(img_p[-1]) is None:
            self.slide_objs[img_p[-1]] = OpenSlide(img_p[0])

        slide_obj = self.slide_objs[img_p[-1]]

        # calculate sample coordinates
        y_c = int(patch_coords[0] * self._lookup_f + 0.5*self._lookup_f)
        x_c = int(patch_coords[1] * self._lookup_f + 0.5*self._lookup_f)

        # sample seg and patch
        sampled = sample(slide_obj, img_p[1], img_p[2], self._patch_size, self._base_scale, (y_c, x_c), self._transforms)
        img = sampled[0]
        seg = sampled[1]

        # sample style input
        style_img_num = int(img_num/12) #TODO un-hardcode this
        style_list = self.style_samp_list[style_img_num]

        style_imgs = self._style_sampler.sample_imgs(slide_obj, (y_c, x_c), img_p[1], self._patch_size, self._base_scale, style_list, self.kernel_size)

        one_hot = F.one_hot(seg.to(torch.long), num_classes=self._num_classes).permute(2,0,1).to(torch.float32)

        # randomly drop style
        if np.random.uniform(0, 1.0) < self._style_drop_rate:
            style_imgs = torch.zeros_like(style_imgs)-0.5

        return img*2-1, one_hot, seg, style_imgs*2-1


class HER2_DS_Predict(HER2_DS_Anno):
    def __getitem__(self, idx):
        return *super().__getitem__(idx), idx


class HER2_DS_Ordered(torch.utils.data.Dataset):
    def __init__(self, img_l, patch_size, base_scale, overlap, num_classes, transforms=None):
        self._img_l = img_l
        self._patch_size = patch_size
        self._base_scale = base_scale
        self._overlap = overlap
        self._num_classes = num_classes
        self._transforms = transforms

        self.slide_objs = {}

        if len(img_l) > 0:
            ref_img = cv2.imdecode(self._img_l[0][2], cv2.IMREAD_ANYDEPTH)
        else:
            ref_img = np.zeros((0,0))

        self._patcher = PatchHandlerMS(ref_img.shape, int(patch_size*base_scale), int(overlap*base_scale))
        self._img_p_num = self._patcher.num_segs()

        unique_wsi = np.unique(np.array([img_tup[-1] for img_tup in self._img_l]))
        self.wsi_lookup = {}
        for num, wsi_num in enumerate(unique_wsi):
            self.wsi_lookup[wsi_num] = num

    def __len__(self):
        num = self._img_p_num * len(self._img_l)
        return num
  
    def __getitem__(self, idx):
        img_num = int(idx/self._img_p_num)
        p_num = idx % self._img_p_num

        img_p = self._img_l[img_num]

        if self.slide_objs.get(img_p[-1]) is None:
            self.slide_objs[img_p[-1]] = OpenSlide(img_p[0])

        slide_obj = self.slide_objs[img_p[-1]]

        seg = cv2.imdecode(img_p[2], cv2.IMREAD_ANYDEPTH)
        seg_crop = self._patcher.get(seg, p_num, 1)

        kernel_size = (self._patch_size - 2*self._overlap) * self._base_scale
        j = int(p_num / self._patcher._x_p)
        i = int(p_num % self._patcher._x_p)

        x_c = int(i * kernel_size + 0.5*kernel_size)
        y_c = int(j * kernel_size + 0.5*kernel_size)

        cut_x = 0
        cut_y = 0

        if (x_c + 0.5*kernel_size) >= seg.shape[1]:
            cut_x = (x_c + 0.5*kernel_size) - seg.shape[1]
            cut_x = int(cut_x / self._base_scale)
        
        if (y_c + 0.5*kernel_size) >= seg.shape[0]:
            cut_y = (y_c + 0.5*kernel_size) - seg.shape[0]
            cut_y = int(cut_y / self._base_scale)

        img_crop = wsi_sample(slide_obj, img_p[1], self._patch_size, self._base_scale, (y_c, x_c))

        applied = self._transforms(image=img_crop, mask=seg_crop)
        img = applied['image']
        seg = applied['mask']
        one_hot = F.one_hot(seg.to(torch.long), num_classes=self._num_classes).permute(2,0,1).to(torch.float32)

        return img*2-1, one_hot, seg, (cut_y, cut_x), self.wsi_lookup[img_p[-1]]


class HER2_DS_UnAnno(torch.utils.data.Dataset):
    def __init__(self, wsi_pd, samples, patch_size, base_scale, overlap, num_classes, style_sampler, transforms=None):
        self._wsi_pd = wsi_pd
        self._samples = samples
        self._patch_size = patch_size
        self._base_scale = base_scale
        self._overlap = overlap
        self._num_classes = num_classes
        self._style_sampler = style_sampler
        self._transforms = transforms

        self.kernel_size = (patch_size - 2*overlap) * self._base_scale
        
        self.slide_info = []
        self._ds_list = []

        self._sample_list = []
        self._sample_ref = []
        self._patch_total = 0

        # create list for all patch numbers, OpenSlide objects, scores
        print("Setting up HER2 unannotated dataset")
        sys.stdout.flush()
        for idx, el in tqdm(self._wsi_pd.iterrows(), desc="Processing HER2 WSI", total=len(self._wsi_pd)):
            slide = OpenSlide(el["File Path"])

            ref_img = slide.read_region(location=(0,0), level=2, size=slide.level_dimensions[2])
            ref_img = np.min(np.array(ref_img)[:,:,:3], axis=2)
            sample_mask = downscale_local_mean(ref_img, (int(self.kernel_size / 16), int(self.kernel_size / 16)), cval=255)<253
            sample_mask = binary_dilation(sample_mask, iterations=2)
            sample_mask = binary_erosion(sample_mask, iterations=2)
            sample_mask = binary_opening(sample_mask, iterations=2)

            coords = np.argwhere(sample_mask)
            coords = np.concatenate((coords, np.full( (len(coords),1), idx, dtype=coords.dtype)), axis=1)

            self._sample_list.append(coords)
            self._sample_ref.append(list(range(self._patch_total, self._patch_total + len(coords))))
            self._patch_total += len(coords)

            dim = (int(slide.dimensions[1]/self._base_scale), int(slide.dimensions[0]/self._base_scale))

            self._ds_list.append((el["File Path"], dim, (sample_mask.shape[1], sample_mask.shape[0]), el["HER2 Score"], el["Slide ID"]))
            self.slide_info.append((str(el["Slide ID"]), dim, sample_mask.shape[1], sample_mask.shape[0], el["HER2 Score"]))

            slide.close()

        self.slide_info = pd.DataFrame(self.slide_info, columns=["Slide ID", "Slide Dim", "Patches x", "Patches y", "HER2 Score"])
        self._ds_list = pd.DataFrame(self._ds_list, columns=["File Path", "Slide Dim", "Patch Dim", "HER2 Score", "Slide ID"])

        self.slide_obj = None
        self.curr_slide = None


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
        img_p = self._ds_list.iloc[img_num]

        # create openslide object
        if self.curr_slide is None:
            self.slide_obj = OpenSlide(img_p["File Path"])
            self.curr_slide = img_p["Slide ID"]
        elif self.curr_slide != img_p["Slide ID"]:
            self.slide_obj.close()
            self.slide_obj = OpenSlide(img_p["File Path"])
            self.curr_slide = img_p["Slide ID"]

        slide_obj = self.slide_obj

        y_c = (patch_coords[0] + 0.5) * self.kernel_size
        x_c = (patch_coords[1] + 0.5) * self.kernel_size
        
        img_crop = wsi_sample(slide_obj, (0,0), self._patch_size, self._base_scale, (y_c, x_c))

        applied = self._transforms(image=img_crop)
        img = applied['image']
        seg = np.full((self._patch_size, self._patch_size), -1, dtype=np.int32)

        # sample style input
        style_imgs = self._style_sampler.sample_imgs(slide_obj, (y_c, x_c), (0,0), self._patch_size, self._base_scale, instance_list, self.kernel_size)

        one_hot = torch.zeros((self._num_classes, self._patch_size, self._patch_size), dtype=torch.float32)

        return img*2-1, one_hot, seg, style_imgs*2-1, sampled_instance


class HER2_DS_Syn(torch.utils.data.Dataset):
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


class HER2_DS_Syn_Ordered(torch.utils.data.Dataset):
    def __init__(self, img_l, num_classes, transforms=None):
        self._img_l = img_l
        self._num_classes = num_classes
        self._transforms = transforms


    def __len__(self):
        return len(self._img_l)


    def __getitem__(self, idx):
        img_tup = self._img_l[idx]

        img = np.array(Image.open(img_tup[0]))
        seg = np.array(Image.open(img_tup[1]).convert('L'))

        applied = self._transforms(image=img, mask=seg)
        img = applied['image']
        seg = applied['mask']
        one_hot = F.one_hot(seg.to(torch.long), num_classes=self._num_classes).permute(2,0,1).to(torch.float32)

        return img*2-1, one_hot, seg, (0, 0), idx
