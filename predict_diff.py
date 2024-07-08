# on windows, openslide needs to be installed
# but even then, we have to manually tell the system where to find it
import os
if os.name == 'nt':
    os.add_dll_directory("C:/Program Files/OpenSlide/bin")

# disable libtiff warnings, which happen for some WSI formats
import ctypes
if os.name == 'nt':
    libtiff = ctypes.cdll.LoadLibrary("libtiff-5.dll")
else:
    import ctypes.util
    lib = ctypes.util.find_library('tiff')
    libtiff = ctypes.cdll.LoadLibrary(lib)

libtiff.TIFFSetWarningHandler(0)


import hydra
import torch
import pytorch_lightning as pl

from pathlib import Path
from datetime import timedelta
from omegaconf import DictConfig
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.strategies.ddp import DDPStrategy

from modules.ldm_diffusion import LDM_Diffusion
from data.dm import DataModule



@hydra.main(version_base=None, config_path="conf", config_name="config_predict")
def main(cfg : DictConfig):
    # calculate batch_size
    cfg.data.batch_size = int(cfg.data.batch_base * cfg.location.batch_mul)
    # create ckpt path
    if hasattr(cfg, "ckpt_name"):
        ckpt_name = cfg.ckpt_name
    else:
        ckpt_name = f"Diff_{cfg.data.name}_{cfg.data.class_train_samples}_{cfg.style_sampling.name}_last.ckpt"
    ckpt_path = cfg.location.result_dir + "/checkpoints/" + ckpt_name

    # delete pretrained ckpt path
    del cfg.diffusion.ckpt_path
    # load module
    module = LDM_Diffusion.load_from_checkpoint(ckpt_path, cfg=cfg, map_location=torch.device("cpu"), strict=False)

    # create predict output directory
    if hasattr(cfg, "predict_dir"):
        predict_dir = cfg.location.data_dir + "/syn_data/" + cfg.predict_dir
    else:
        predict_dir = cfg.location.data_dir + "/syn_data/" + f"{cfg.data.name}_{cfg.data.class_train_samples}_{cfg.style_sampling.name}_cfg{cfg.cfg_scale}"


    Path(predict_dir).mkdir(parents=True, exist_ok=True)
    # set predict output directory
    module.predict_dir = predict_dir

    # double the number of created images
    cfg.data.samples = 2 * cfg.data.samples

    # create data module
    data_module = DataModule(cfg)
    
    # set float point precision
    torch.set_float32_matmul_precision('high')

    # compile code if requested (pytorch 2.0)
    if cfg.location.compile == 'disable':
        module = module
    elif cfg.location.compile == 'default':
        module = torch.compile(module)
    elif cfg.location.compile == 'reduce-overhead':
        module = torch.compile(module, mode="reduce-overhead")
    elif cfg.location.compile == 'max-autotune':
        module = torch.compile(module, mode="max-autotune")


    progress_bar = TQDMProgressBar(refresh_rate=int(256/cfg.data.batch_size))
    callbacks = [progress_bar] # more callbacks can be added

    trainer = pl.Trainer(max_epochs=1, callbacks=callbacks,
                         accelerator='gpu', devices=cfg.location.n_gpus,
                         strategy=DDPStrategy(find_unused_parameters=True, process_group_backend=cfg.location.backend, timeout=timedelta(seconds=7200*2)),
                         accumulate_grad_batches=4, num_sanity_val_steps=0)

    trainer.predict(module, data_module)


if __name__ == "__main__":
    main()