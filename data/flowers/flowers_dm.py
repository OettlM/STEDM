import os
import sys
import cv2
import shutil
import scipy.io

import albumentations as A
import pytorch_lightning as pl

from pathlib import Path
from torch.utils.data import DataLoader
from albumentations.pytorch.transforms import ToTensorV2

from data.flowers.flowers_ds import Flowers_DS, Flowers_DS_Predict, Flowers_DS_Ordered, Flowers_DS_Syn, Flowers_DS_Syn_Ordered

from ldm.util import get_obj_from_str




class Flowers_DM_Anno(pl.LightningDataModule):
    def __init__(self, cfg, ds_cfg, ratio, **kwargs):
        super().__init__()
        # store configs
        self._cfg = cfg
        self._ds_cfg = ds_cfg

        # location based parameters
        self._data_dir = cfg.location.data_dir
        self._n_workers = cfg.location.n_workers
        self._location = cfg.location.name
        # basic data parameters
        self._batch_size = cfg.data.batch_size
        self._patch_size = cfg.data.patch_size
        self._num_classes = cfg.data.num_classes
        # sample number for this dataloader
        self._samples = int(cfg.data.samples * ratio)
        # data storage parameters
        self._reload_data = cfg.data.reload_data
        self._zip_name = ds_cfg.zip_name + "_" + ds_cfg.annotator.name  if hasattr(ds_cfg, "annotator") else ds_cfg.zip_name


    def prepare_data(self):
        base_dir = self._data_dir + "/" + self._zip_name
        zip_file = self._data_dir + "/" + self._zip_name + ".zip"
        
        # refresh data, load wsi, load annos, create label mask, create zip
        if self._reload_data or not os.path.isfile(zip_file):
            shutil.make_archive(zip_file[:-4], 'zip', base_dir)
        
        if self._location == "pc":
            if not os.path.isdir(base_dir):
                shutil.unpack_archive(zip_file, base_dir)
        else:
            local_dir = os.path.join('/scratch', os.environ['SLURM_JOB_ID'])
            Path(local_dir).mkdir(parents=True, exist_ok=True)
            local_zip_file = local_dir + "/" + self._zip_name + ".zip"
            print(f"Copy {self._zip_name} zip file")
            sys.stdout.flush()
            shutil.copyfile(zip_file, local_zip_file)
            print(f"Unpack {self._zip_name} zip file")
            sys.stdout.flush()
            shutil.unpack_archive(local_zip_file, local_dir + "/" + self._zip_name)
            print(f"Delete {self._zip_name} zip file")
            sys.stdout.flush()
            os.remove(local_zip_file)
            print(f"Finished {self._zip_name} preparation")
            sys.stdout.flush()


    def setup(self, stage):
        if self._location == "pc":
            base_dir = self._data_dir + "/" + self._zip_name
        else:
            base_dir = os.path.join('/scratch', os.environ['SLURM_JOB_ID']) + "/" + self._zip_name

        # load labels file
        labels = scipy.io.loadmat(base_dir + "/imagelabels.mat")['labels'][0]
        splits = scipy.io.loadmat(base_dir + "/setid.mat")

        train_idx = splits['trnid'][0]
        val_idx = splits['valid'][0]
        test_idx = splits['tstid'][0]

        # split images category wise
        self._list_train = []
        self._list_val = []
        self._list_test = []
        self._list_unanno = []

        label_bin_list = [[] for i in range(102)]

        for i, label in enumerate(labels):
            el = (base_dir+f"/imgs/image_{str(i+1).zfill(5)}.jpg", base_dir+f"/segs/segmim_{str(i+1).zfill(5)}.jpg", label-1, i+1)

            if (i+1) in train_idx:
                label_bin_list[label-1].append(el)
            elif (i+1) in val_idx:
                self._list_val.append(el)
            elif (i+1) in test_idx:
                self._list_test.append(el)
            else:
                raise Exception("Element could not be assigned to train/val/test!")

        # different splits
        for label, bin_list in enumerate(label_bin_list):
            if label in self._ds_cfg.train_classes:
                self._list_train.extend(bin_list[:self._cfg.data.class_train_samples])
                self._list_unanno.extend(bin_list[self._cfg.data.class_train_samples:])
            elif label in self._ds_cfg.unanno_classes:
                self._list_unanno.extend(bin_list)
            else:
                raise Exception("Class not assigned!")

        # setup augments
        base_transforms = A.Compose([A.Resize(self._patch_size, self._patch_size), A.HorizontalFlip(), A.VerticalFlip(),
                                      A.ToFloat(max_value=255), ToTensorV2()])

        val_test_transforms = A.Compose([A.Resize(self._patch_size, self._patch_size), A.ToFloat(max_value=255), ToTensorV2()])

        style_transforms = A.Compose([A.Resize(self._patch_size, self._patch_size), A.HorizontalFlip(), A.VerticalFlip(),
                             A.Affine(scale=(0.8, 1.2), translate_percent=0.1, rotate=(-360,360), shear=(-20,20), mode=cv2.BORDER_REFLECT, p=1.0),
                             A.ToFloat(max_value=255), ToTensorV2()])

        # setup style sampler class
        style_sampler_class = get_obj_from_str("data.flowers.style_sampler." + self._cfg.style_sampling.class_name)
        style_sampler = style_sampler_class(self._cfg.style_sampling, style_transforms)
        style_sampler_pred = style_sampler_class(self._cfg.style_sampling, style_transforms)


        # get style drop rate
        style_drop_rate = self._cfg.style_drop_rate if hasattr(self._cfg, "style_drop_rate") else 0.0

        # create datasets
        self._ds_train = Flowers_DS(self._list_train, self._samples, self._num_classes, base_transforms, style_sampler, style_drop_rate)
        self._ds_val = Flowers_DS_Ordered(self._list_val, self._num_classes, val_test_transforms)
        self._ds_test = Flowers_DS_Ordered(self._list_test, self._num_classes, val_test_transforms)
        self._ds_predict = Flowers_DS_Predict(self._list_train, self._samples, self._num_classes, base_transforms, style_sampler_pred, 0.0)


    def train_dataloader(self):
        return DataLoader(self._ds_train, batch_size=self._batch_size, num_workers=self._n_workers, pin_memory=True, prefetch_factor=2, shuffle=True, persistent_workers=True)
   
    def val_dataloader(self):
        return DataLoader(self._ds_val, batch_size=self._batch_size, num_workers=self._n_workers, pin_memory=True, prefetch_factor=2, shuffle=True, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self._ds_test, batch_size=self._batch_size, num_workers=self._n_workers, pin_memory=True, prefetch_factor=2, shuffle=False, persistent_workers=True)

    def predict_dataloader(self):
        return DataLoader(self._ds_predict, batch_size=self._batch_size, num_workers=self._n_workers, pin_memory=True, prefetch_factor=2, shuffle=False, persistent_workers=True)

    def train_dataset(self):
        return self._ds_train

    def val_dataset(self):
        return self._ds_val

    def test_dataset(self):
        return self._ds_test

    def predict_dataset(self):
        return self._ds_predict


class Flowers_DM_UnAnno(Flowers_DM_Anno):
        def setup(self, stage):
            if self._location == "pc":
                base_dir = self._data_dir + "/" + self._zip_name
            else:
                base_dir = os.path.join('/scratch', os.environ['SLURM_JOB_ID']) + "/" + self._zip_name

            # load labels file
            labels = scipy.io.loadmat(base_dir + "/imagelabels.mat")['labels'][0]
            splits = scipy.io.loadmat(base_dir + "/setid.mat")

            train_idx = splits['trnid'][0]
            val_idx = splits['valid'][0]
            test_idx = splits['tstid'][0]

            # split images category wise
            self._list_train = []
            self._list_val = []
            self._list_test = []
            self._list_unanno = []

            label_bin_list = [[] for i in range(102)]

            for i, label in enumerate(labels):
                el = (base_dir+f"/imgs/image_{str(i+1).zfill(5)}.jpg", base_dir+f"/segs/segmim_{str(i+1).zfill(5)}.jpg", label-1)

                if (i+1) in train_idx:
                    label_bin_list[label-1].append(el)
                elif (i+1) in val_idx:
                    self._list_val.append(el)
                elif (i+1) in test_idx:
                    self._list_test.append(el)
                else:
                    raise Exception("Element could not be assigned to train/val/test!")

            # differnt splits
            train_bin_list = label_bin_list[:20]
            unanno_bin_list = label_bin_list[20:]

            for bin in train_bin_list:
                self._list_train.extend(bin[:self._cfg.data.class_train_samples])
                self._list_unanno.extend(bin[self._cfg.data.class_train_samples:])
            for bin in unanno_bin_list:
                self._list_unanno.extend(bin)
            
            # setup augments
            base_transforms = A.Compose([A.Resize(self._patch_size, self._patch_size), A.HorizontalFlip(), A.VerticalFlip(),
                                        A.ToFloat(max_value=255), ToTensorV2()])

            style_transforms = A.Compose([A.Resize(self._patch_size, self._patch_size), A.HorizontalFlip(), A.VerticalFlip(),
                                A.Affine(scale=(0.8, 1.2), translate_percent=0.1, rotate=(-360,360), shear=(-20,20), mode=cv2.BORDER_REFLECT),
                                A.ToFloat(max_value=255), ToTensorV2()])

            # setup style sampler class
            style_sampler_class = get_obj_from_str("data.flowers.style_sampler." + self._cfg.style_sampling.class_name)
            style_sampler_pred = style_sampler_class(self._cfg.style_sampling, style_transforms)

            # create datasets
            self._ds_train = []
            self._ds_val = []
            self._ds_test = []
            self._ds_predict = Flowers_DS(self._list_train, self._samples, self._num_classes, base_transforms, style_sampler_pred, 0.0)


class Flowers_DM_Syn(pl.LightningDataModule):
    def __init__(self, cfg, ds_cfg, ratio, **kwargs):
        super().__init__()
        # store configs
        self._cfg = cfg
        self._ds_cfg = ds_cfg

        # location based parameters
        self._data_dir = cfg.location.data_dir
        self._n_workers = cfg.location.n_workers
        self._location = cfg.location.name
        # basic data parameters
        self._batch_size = cfg.data.batch_size
        self._patch_size = cfg.data.patch_size
        self._overlap = cfg.data.overlap_val
        self._num_classes = cfg.data.num_classes
        # sample number for this dataloader
        self._samples = int(cfg.data.samples * ratio)
        # data storage parameters
        self._reload_data = cfg.data.reload_data

        # create syn folder name
        self._folder_name = ds_cfg.folder_name


    def prepare_data(self):
        syn_path = self._data_dir + "/syn_data/" + self._folder_name
        zip_file = self._data_dir + "/syn_data/" + self._folder_name + ".zip"
        # refresh data, create zip
        if self._reload_data or not os.path.isfile(zip_file):
            shutil.make_archive(syn_path, 'zip', syn_path)
        
        if self._location == "pc":
            if not os.path.isdir(syn_path):
                shutil.unpack_archive(zip_file, syn_path)
        else:
            local_dir = os.path.join('/scratch', os.environ['SLURM_JOB_ID'])
            Path(local_dir).mkdir(parents=True, exist_ok=True)
            local_zip_file = local_dir + "/" + self._folder_name + ".zip"
            print(f"Copy {self._folder_name} zip file")
            sys.stdout.flush()
            shutil.copyfile(zip_file, local_zip_file)
            print(f"Unpack {self._folder_name} zip file")
            sys.stdout.flush()
            shutil.unpack_archive(local_zip_file, local_dir + "/" + self._folder_name)
            print(f"Delete {self._folder_name} zip file")
            sys.stdout.flush()
            os.remove(local_zip_file)
            print(f"Finished {self._folder_name} preparation")
            sys.stdout.flush()


    def setup(self, stage):
        if self._location == "pc":
            base_dir = self._data_dir + "/syn_data/" + self._folder_name
        else:
            base_dir = os.path.join('/scratch', os.environ['SLURM_JOB_ID']) + "/" + self._folder_name

        # create list of files, to potentially split later
        num_samples = int(len(os.listdir(base_dir)) / 2)
        self._list_train = [(base_dir + f"/img_{str(i).zfill(5)}.png", base_dir + f"/seg_{str(i).zfill(5)}.png") for i in range(num_samples)]

        # setup transforms
        aug_train = A.Compose([A.Resize(self._patch_size, self._patch_size),
                               A.HorizontalFlip(), A.VerticalFlip(),
                               A.ToFloat(max_value=255), ToTensorV2()])

        self._ds_train = Flowers_DS_Syn(self._list_train, self._samples, self._num_classes, aug_train)
        self._ds_val = []
        self._ds_test = []
        self._ds_predict = []

        aug_unmod = A.Compose([A.ToFloat(max_value=255), A.Resize(self._patch_size, self._patch_size), ToTensorV2()])
        self._ds_ordered = Flowers_DS_Syn_Ordered(self._list_train, self._num_classes, aug_unmod)


    def train_dataloader(self):
        return DataLoader(self._ds_train, batch_size=self._batch_size, num_workers=self._n_workers, pin_memory=True, prefetch_factor=2, shuffle=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self._ds_val, batch_size=self._batch_size, num_workers=self._n_workers, pin_memory=True, prefetch_factor=2, shuffle=False, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self._ds_test, batch_size=self._batch_size, num_workers=self._n_workers, pin_memory=True, prefetch_factor=2, shuffle=False, persistent_workers=True)

    def predict_dataloader(self):
        return DataLoader(self._ds_predict, batch_size=self._batch_size, num_workers=self._n_workers, pin_memory=True, prefetch_factor=2, shuffle=False, persistent_workers=True)

    def train_dataset(self):
        return self._ds_train

    def val_dataset(self):
        return self._ds_val

    def test_dataset(self):
        return self._ds_test

    def predict_dataset(self):
        return self._ds_predict