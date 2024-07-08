import torch
from einops import rearrange



class Agg_Linear(torch.nn.Module):
    def __init__(self, sampling_cfg, embedder):
        super().__init__()

        self._sampling_cfg = sampling_cfg
        self._embedder = embedder

        num = self._sampling_cfg.num_patches if self._sampling_cfg.name == "mp" else 1

        self._linear_block = torch.nn.Sequential(torch.nn.ReLU(),
                                                 torch.nn.Linear(512*num,512),
                                                 torch.nn.ReLU(),
                                                 torch.nn.Linear(512,512),
                                                 torch.nn.ReLU())

        self.register_module("embedder", self._embedder)
        self.register_module("linear_block", self._linear_block)

    def forward(self, style_imgs):
        b = style_imgs.shape[0]
        style_imgs = rearrange(style_imgs, 'b n h w c -> (b n) c h w')
        
        style_features = self._embedder(style_imgs)
        style_features = rearrange(style_features, '(b1 n) f -> b1 (n f)', b1=b)

        style_features = self._linear_block(style_features)

        return style_features


class Agg_Max(torch.nn.Module):
    def __init__(self, sampling_cfg, embedder):
        super().__init__()

        self._sampling_cfg = sampling_cfg
        self._embedder = embedder

        self.register_module("embedder", self._embedder)
    
    def forward(self, style_imgs):
        b = style_imgs.shape[0]
        style_imgs = rearrange(style_imgs, 'b n h w c -> (b n) c h w')
        
        style_features = self._embedder(style_imgs)
        style_features = rearrange(style_features, '(b1 n) f -> b1 n f', b1=b)

        style_features = torch.max(style_features, dim=1)[0]

        return style_features


class Agg_Mean(torch.nn.Module):
    def __init__(self, sampling_cfg, embedder):
        super().__init__()

        self._sampling_cfg = sampling_cfg
        self._embedder = embedder

        self.register_module("embedder", self._embedder)
    
    def forward(self, style_imgs):
        b = style_imgs.shape[0]
        style_imgs = rearrange(style_imgs, 'b n h w c -> (b n) c h w')
        
        style_features = self._embedder(style_imgs)
        style_features = rearrange(style_features, '(b1 n) f -> b1 n f', b1=b)

        style_features = torch.mean(style_features, dim=1)

        return style_features


class Agg_None(torch.nn.Module):
    def __init__(self, sampling_cfg, embedder):
        super().__init__()

        self._sampling_cfg = sampling_cfg
        self._embedder = embedder

    def forward(self, style_imgs):
        return torch.zeros((style_imgs.shape[0], 512), dtype=style_imgs.dtype, device=style_imgs.device)