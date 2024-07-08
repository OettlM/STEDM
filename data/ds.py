import torch




class Predict_DS(torch.utils.data.Dataset):
    def __init__(self, img_ds, style_ds):
        super().__init__()

        self._imgs_ds = img_ds
        self._style_ds = style_ds


    def __len__(self):
        return len(self._imgs_ds)


    def __getitem__(self, idx):
        imgs_tup = self._imgs_ds[idx]
        style_tup = self._style_ds[idx]

        return imgs_tup[0], imgs_tup[1], imgs_tup[2], style_tup[3], imgs_tup[4]