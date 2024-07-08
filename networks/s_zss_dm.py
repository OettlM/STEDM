import torch
import torchvision

from ldm.models.diffusion.ddpm import LatentDiffusion
from networks.agg_blocks import Agg_None, Agg_Linear, Agg_Max, Agg_Mean
from networks.vit_set import sViT




class S_ZSS_DM(LatentDiffusion):
    def __init__(self, encoder, sampling_cfg, agg_cfg, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sampling_cfg = sampling_cfg
        self._agg_cfg = agg_cfg
        self._cfg = cfg

        self.embed_key = "style_imgs"
        embedder = torchvision.models.get_model(encoder)
        embedder.head = torch.nn.Linear(768, 512)

        if self._sampling_cfg.name == "none":
            self._agg_block = Agg_None(self._sampling_cfg, embedder)
        else:
            if self._agg_cfg.name == "linear":
                self._agg_block = Agg_Linear(self._sampling_cfg, embedder)
            elif self._agg_cfg.name == "max":
                self._agg_block = Agg_Max(self._sampling_cfg, embedder)
            elif self._agg_cfg.name == "mean":
                self._agg_block = Agg_Mean(self._sampling_cfg, embedder)
            elif self._agg_cfg.name == "svit":
                args = dict(self._agg_cfg)
                del args["name"]
                self._agg_block = sViT(image_size=self._cfg.data.patch_size, 
                                        num_classes=512,
                                        ns=self._sampling_cfg.num_patches if self._sampling_cfg.name == "mp" else 1,
                                        **args
                                        )
            else:
                raise Exception("Unkown aggregation function!")

        self.register_module("agg_block", self._agg_block)


    def get_input(self, batch, k, cond_key=None, bs=None, **kwargs):
        self.cond_stage_trainable = True
        outputs = LatentDiffusion.get_input(self, batch, k, bs=bs, **kwargs)
        self.cond_stage_trainable = False

        z, c = outputs[0], outputs[1]
        c = self.get_learned_conditioning(c)

        style_imgs = batch[self.embed_key][:bs]

        style_features = self._agg_block(style_imgs)

        all_conds = {"c_concat": [c], "c_crossattn": [style_features]}
        noutputs = [z, all_conds]
        noutputs.extend(outputs[2:])
        return noutputs