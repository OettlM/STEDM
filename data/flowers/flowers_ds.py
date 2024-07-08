import torch

import numpy as np
import torch.nn.functional as F

from PIL import Image


class Flowers_DS(torch.utils.data.Dataset):
    def __init__(self, img_l, samples, num_classes, base_transforms, style_sampler, style_drop_rate):
        self._img_l = img_l
        self._samples = samples
        self._num_classes = num_classes
        self._base_transforms = base_transforms
        self._style_sampler = style_sampler
        self._style_drop_rate = style_drop_rate

        # create lookup for style sampling
        unique_labels = np.unique(np.array([el[2] for el in self._img_l]))
        self._lookup_dict = {}
        for num, label in enumerate(unique_labels):
            self._lookup_dict[label] = num

        self._style_list = [[] for i in range(len(unique_labels))]

        for el in self._img_l:
            self._style_list[self._lookup_dict[el[2]]].append(el)


    def __len__(self):
        if len(self._img_l) > 0:
            return self._samples
        else:
            return 0


    def __getitem__(self, idx):
        # sample image
        sampled_img = np.random.randint(0, len(self._img_l))

        # get requested image
        img_tup = self._img_l[sampled_img]

        # load image
        img_raw = np.array(Image.open(img_tup[0]))
        
        # load and process segmenation
        seg = np.array(Image.open(img_tup[1]))
        seg = (1-(seg[:,:,0]<=10)*(seg[:,:,1]<=10)*(seg[:,:,2]>=220)).astype(np.uint8)

        # apply augmentations
        applied = self._base_transforms(image=img_raw, mask=seg)
        img = applied['image']
        seg = applied['mask']

        # get one_hot segmenation
        one_hot = F.one_hot(seg.to(torch.long), num_classes=self._num_classes).permute(2,0,1).to(torch.float32)

        style_imgs = self._style_sampler.sample_imgs(img_raw)

        # randomly drop style
        if np.random.uniform(0, 1.0) < self._style_drop_rate:
            style_imgs = torch.zeros_like(style_imgs)-0.5

        # adjust data range and return
        return img*2-1, one_hot, seg, style_imgs*2-1


class Flowers_DS_Predict(Flowers_DS):
    def __getitem__(self, idx):
        return *super().__getitem__(idx), idx


class Flowers_DS_Ordered(torch.utils.data.Dataset):
    def __init__(self, img_l, num_classes, base_transforms):
        self._img_l = img_l
        self._num_classes = num_classes
        self._base_transforms = base_transforms

        unique_wsi = np.unique(np.array([img_tup[-1] for img_tup in self._img_l]))
        self.wsi_lookup = {}
        for num, wsi_num in enumerate(unique_wsi):
            self.wsi_lookup[wsi_num] = num


    def __len__(self):
        return len(self._img_l)


    def __getitem__(self, idx):
        # get requested image
        img_tup = self._img_l[idx]

        # load image
        img_raw = np.array(Image.open(img_tup[0]))
        
        # load and process segmenation
        seg = np.array(Image.open(img_tup[1]))
        seg = (1-(seg[:,:,0]<=10)*(seg[:,:,1]<=10)*(seg[:,:,2]>=220)).astype(np.uint8)

        # apply augmentations
        applied = self._base_transforms(image=img_raw, mask=seg)
        img = applied['image']
        seg = applied['mask']

        # get one_hot segmenation
        one_hot = F.one_hot(seg.to(torch.long), num_classes=self._num_classes).permute(2,0,1).to(torch.float32)

        # adjust data range and return
        return img*2-1, one_hot, seg, (0, 0), self.wsi_lookup[img_tup[-1]]


class Flowers_DS_Syn(torch.utils.data.Dataset):
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


class Flowers_DS_Syn_Ordered(torch.utils.data.Dataset):
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