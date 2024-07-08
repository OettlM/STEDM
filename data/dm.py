import pytorch_lightning as pl

from torch.utils.data import DataLoader, ConcatDataset

from data.ds import Predict_DS
from ldm.util import get_obj_from_str



class DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg

        self._batch_size = cfg.data.batch_size
        self._n_workers = cfg.location.n_workers

        self._data_modules = {}

        self.init_data_modules()

        self._is_prepared = False


    def init_data_modules(self):
        for ds_name, ratio in zip(self._cfg.data.dataset, self._cfg.data.ratios):
            if ratio > 0.0:
                ds_config = self._cfg.data.dataset[ds_name]

                dm_class = get_obj_from_str(ds_config.file)
                dm = dm_class(self._cfg, ds_config, ratio)

                self._data_modules[ds_name] = dm


    def prepare_data(self):
        if not self._is_prepared:
            for data_module in self._data_modules.values():
                data_module.prepare_data()
            
            self._is_prepared = True


    def setup(self, stage):
        for data_module in self._data_modules.values():
            data_module.setup(stage)


    def train_dataloader(self):
        train_datasets = []
        for data_module in self._data_modules.values():
            dataset = data_module.train_dataset()
            if len(dataset) > 0:
                train_datasets.append(dataset)
        
        comb_train_dataset = ConcatDataset(train_datasets)
        return DataLoader(comb_train_dataset, batch_size=self._batch_size, num_workers=self._n_workers, pin_memory=True, prefetch_factor=2, shuffle=True, persistent_workers=True)


    def val_dataloader(self):
        val_datasets = []
        for data_module in self._data_modules.values():
            dataset = data_module.val_dataset()
            if len(dataset) > 0:
                val_datasets.append(dataset)
        
        comb_val_dataset = ConcatDataset(val_datasets)
        return DataLoader(comb_val_dataset, batch_size=self._batch_size, num_workers=self._n_workers, pin_memory=True, prefetch_factor=2, shuffle=True, persistent_workers=True)


    def test_dataloader(self):
        test_datasets = []
        for data_module in self._data_modules.values():
            dataset = data_module.test_dataset()
            if len(dataset) > 0:
                test_datasets.append(dataset)
        
        comb_test_dataset = ConcatDataset(test_datasets)
        return DataLoader(comb_test_dataset, batch_size=self._batch_size, num_workers=self._n_workers, pin_memory=True, prefetch_factor=2, shuffle=False, persistent_workers=True)


    def predict_dataloader(self):
        img_ds = self._data_modules[list(self._cfg.data.dataset.keys())[0]].predict_dataset()
        style_ds = self._data_modules[list(self._cfg.data.dataset.keys())[1]].predict_dataset()

        comb_predict_dataset = Predict_DS(img_ds, style_ds)
        return DataLoader(comb_predict_dataset, batch_size=self._batch_size, num_workers=self._n_workers, pin_memory=True, prefetch_factor=2, shuffle=False, persistent_workers=True)