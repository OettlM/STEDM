import os
import cv2
import sys
import h5py
import shutil
import pickle

import pandas as pd
import albumentations as A
import pytorch_lightning as pl

from pathlib import Path
from torch.utils.data import DataLoader
from albumentations.pytorch.transforms import ToTensorV2

from data.catch.catch_utils import wsi_anno_exact_multi
from data.catch.catch_ds import CATCH_DS_Anno, CATCH_DS_Predict, CATCH_DS_Ordered, CATCH_DS_UnAnno, CATCH_DS_Syn

from ldm.util import get_obj_from_str




class CATCH_DM_Anno(pl.LightningDataModule):
    def __init__(self, cfg, ds_cfg, ratio, **kwargs):
        super().__init__()
        # store configs
        self._cfg = cfg
        self._ds_cfg = ds_cfg
        self._ratio = ratio

        # location based parameters
        self._data_dir = cfg.location.data_dir
        self._n_workers = cfg.location.n_workers
        self._location = cfg.location.name
        # basic data parameters
        self._batch_size = cfg.data.batch_size
        self._base_scale = cfg.data.base_scale
        self._patch_size = cfg.data.patch_size
        self._overlap = cfg.data.overlap_val
        self._num_classes = cfg.data.num_classes
        self._classes = cfg.data.classes
        # sample number for this dataloader
        self._samples = int(cfg.data.samples * ratio)
        # data storage parameters
        self._reload_data = cfg.data.reload_data
        self._zip_name = ds_cfg.zip_name + "_" + ds_cfg.annotator.name  if hasattr(ds_cfg, "annotator") else ds_cfg.zip_name
        # parameters for creating the segmentations
        self._lookup_f = cfg.data.lookup_f
        self._label_dict = ds_cfg.label_dict

    
    def prepare_data(self):
        base_dir = self._data_dir + "/" + self._zip_name
        zip_file = self._data_dir + "/" + self._zip_name + ".zip"
        
        # refresh data, load wsi, load annos, create label mask, create zip
        if self._reload_data or not os.path.isfile(zip_file):
            # create local dirs to prepare data
            wsi_dir = base_dir + "/wsi"
            Path(base_dir).mkdir(parents=True, exist_ok=True)
            Path(wsi_dir).mkdir(parents=True, exist_ok=True)

            # create exact login tuple
            exact_login = (self._ds_cfg.anno_server.adress, self._ds_cfg.anno_server.user, self._ds_cfg.anno_server.pw)

            # creat anno file
            anno_file = h5py.File(base_dir + "/anno.hdf5", "w")

            # create lists
            list_train = []
            list_val = []
            list_test = []

            # get images and segmentation from server
            wsi_list = wsi_anno_exact_multi(exact_login, self._ds_cfg.img_sets, wsi_dir, self._ds_cfg.anno_product, anno_file, self._label_dict, self._lookup_f, users=self._ds_cfg.annotator.users)
        
            # split data
            if hasattr(self._ds_cfg, "split_file"):
                splits = pd.read_csv(self._data_dir+"/"+self._ds_cfg.split_file, sep=";")
                target_dict = {"train":list_train, "val":list_val, "test":list_test}
                for roi_el in wsi_list:
                    target = splits[splits["Slide"]==roi_el[0]]["Dataset"]
                    target_dict[target.values[0]].append(roi_el)
            else:
                list_train = wsi_list

            # close anno file
            anno_file.close()

            # save computed segmentations
            with open(base_dir + "/train.pkl", "wb") as f:
                pickle.dump(list_train, f)
            with open(base_dir + "/val.pkl", "wb") as f:
                pickle.dump(list_val, f)
            with open(base_dir + "/test.pkl", "wb") as f:
                pickle.dump(list_test, f)

            shutil.make_archive(zip_file[:-4], 'zip', base_dir)

        if self._location == "pc":
            if not os.path.isdir(base_dir):
                shutil.unpack_archive(zip_file, base_dir)
        else:
            local_dir = os.path.join('/scratch', os.environ['SLURM_JOB_ID'])

            if not os.path.isdir(local_dir + "/" + self._zip_name):
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

        with open(base_dir + "/train.pkl", "rb") as f:
            list_train = pickle.load(f)
            self._list_train = [(base_dir + "/wsi/" + el[0], *el[1:]) for el in list_train]

        with open(base_dir + "/val.pkl", "rb") as f:
            list_val = pickle.load(f)
            self._list_val = [(base_dir + "/wsi/" + el[0], *el[1:]) for el in list_val]     

        with open(base_dir + "/test.pkl", "rb") as f:
            list_test = pickle.load(f)
            self._list_test = [(base_dir + "/wsi/" + el[0], *el[1:]) for el in list_test]

        self._anno_path = base_dir + "/anno.hdf5"

        # split into anno and unanno
        classes_lists = [[] for i in range(self._num_classes-1)]

        for el in self._list_train:
            for class_num, class_name in enumerate(self._classes[1:]):
                if class_name in el[0]:
                    classes_lists[class_num].append(el)

        self._list_train = []
        self._list_unanno = []

        for class_list in classes_lists:
            self._list_train.extend(class_list[:self._cfg.data.class_train_samples])
            self._list_unanno.extend(class_list[self._cfg.data.class_train_samples:])

        # setup transforms
        train_transforms = A.Compose([A.Resize(self._patch_size, self._patch_size), A.HorizontalFlip(), A.VerticalFlip(),
                                      A.ToFloat(max_value=255), ToTensorV2()])

        val_test_transforms = A.Compose([A.Resize(self._patch_size, self._patch_size), A.ToFloat(max_value=255), ToTensorV2()])

        predict_transforms = A.Compose([A.Resize(self._patch_size, self._patch_size), A.HorizontalFlip(), A.VerticalFlip(),
                             A.Affine(scale=(0.8, 1.2), translate_percent=0.1, rotate=(-360,360), shear=(-20,20), mode=cv2.BORDER_REFLECT),
                             A.ToFloat(max_value=255), ToTensorV2()])

        # setup style sampler class
        style_sampler_class = get_obj_from_str("data.catch.style_sampler." + self._cfg.style_sampling.class_name)
        style_sampler = style_sampler_class(self._cfg.style_sampling, train_transforms)
        style_sampler_pred = style_sampler_class(self._cfg.style_sampling, predict_transforms)

        # get style drop rate
        style_drop_rate = self._cfg.style_drop_rate if hasattr(self._cfg, "style_drop_rate") else 0.0

        self._ds_train = CATCH_DS_Anno(self._list_train, self._anno_path, self._patch_size, self._base_scale, self._overlap, self._samples, self._lookup_f, self._num_classes, style_sampler, style_drop_rate, train_transforms)
        self._ds_val = CATCH_DS_Ordered(self._list_val, self._anno_path, self._patch_size, self._base_scale, self._overlap, self._lookup_f, 4, self._num_classes, val_test_transforms)
        self._ds_test = CATCH_DS_Ordered(self._list_test, self._anno_path, self._patch_size, self._base_scale, self._overlap, self._lookup_f, 1, self._num_classes, val_test_transforms)
        self._ds_predict = CATCH_DS_Predict(self._list_train, self._anno_path, self._patch_size, self._base_scale, self._overlap, self._samples, self._lookup_f, self._num_classes, style_sampler_pred, 0.0, predict_transforms)


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


class CATCH_DM_UnAnno(CATCH_DM_Anno):
    def setup(self, stage):
        if self._location == "pc":
            base_dir = self._data_dir + "/" + self._zip_name
        else:
            base_dir = os.path.join('/scratch', os.environ['SLURM_JOB_ID']) + "/" + self._zip_name

        with open(base_dir + "/train.pkl", "rb") as f:
            list_train = pickle.load(f)
            self._list_train = [(base_dir + "/wsi/" + el[0], *el[1:]) for el in list_train]

        with open(base_dir + "/val.pkl", "rb") as f:
            list_val = pickle.load(f)
            self._list_val = [(base_dir + "/wsi/" + el[0], *el[1:]) for el in list_val]     

        with open(base_dir + "/test.pkl", "rb") as f:
            list_test = pickle.load(f)
            self._list_test = [(base_dir + "/wsi/" + el[0], *el[1:]) for el in list_test]

        self._anno_path = base_dir + "/anno.hdf5"

        # split into anno and unanno
        classes_lists = [[] for i in range(self._num_classes-1)]

        for el in self._list_train:
            for class_num, class_name in enumerate(self._classes[1:]):
                if class_name in el[0]:
                    classes_lists[class_num].append(el)

        self._list_train = []
        self._list_unanno = []

        for class_list in classes_lists:
            self._list_train.extend(class_list[:self._cfg.data.class_train_samples])
            self._list_unanno.extend(class_list[self._cfg.data.class_train_samples:])

        # setup transforms
        predict_transforms = A.Compose([A.Resize(self._patch_size, self._patch_size), A.HorizontalFlip(), A.VerticalFlip(),
                             A.Affine(scale=(0.8, 1.2), translate_percent=0.1, rotate=(-360,360), shear=(-20,20), mode=cv2.BORDER_REFLECT),
                             A.ToFloat(max_value=255), ToTensorV2()])

        # setup style sampler class
        style_sampler_class = get_obj_from_str("data.catch.style_sampler." + self._cfg.style_sampling.class_name)
        style_sampler_pred = style_sampler_class(self._cfg.style_sampling, predict_transforms)

        self._ds_train = []
        self._ds_val = []
        self._ds_test = []
        self._ds_predict = CATCH_DS_UnAnno(self._list_unanno, self._patch_size, self._base_scale, self._overlap, self._samples, self._num_classes, style_sampler_pred, predict_transforms)


class CATCH_DM_Syn(pl.LightningDataModule):
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
        self._base_scale = cfg.data.base_scale
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

        self._ds_train = CATCH_DS_Syn(self._list_train, self._samples, self._num_classes, aug_train)
        self._ds_val = []
        self._ds_test = []
        self._ds_predict = []


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