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

from data.dm import DataModule
from modules.segmentation import Segmentation

import pytorch_lightning as pl
from datetime import timedelta
from omegaconf import DictConfig
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.strategies.ddp import DDPStrategy



@hydra.main(version_base=None, config_path="conf", config_name="config_seg")
def main(cfg : DictConfig):
    # calculate batch_size
    cfg.data.batch_size = int(cfg.data.batch_base * cfg.location.batch_mul)
    # calculate learning rate
    cfg.lr = cfg.base_lr * cfg.data.batch_size * cfg.location.n_gpus

    if cfg.run_name is not None:
        run_name = cfg.run_name
    else:
        run_name=cfg.data.dataset[f"{cfg.data.name}_syn"].folder_name
        cfg.run_name = run_name

    logger = pl_loggers.WandbLogger(project=f"{cfg.data.name.upper()} Style Seg", name=run_name)

    data_module = DataModule(cfg)

    if hasattr(cfg, "ckpt_name"):
        ckpt_path = cfg.location.result_dir + "/checkpoints/" + cfg.ckpt_name
        module = Segmentation.load_from_checkpoint(ckpt_path, cfg=cfg)
    else:
        module = Segmentation(cfg)

    # set float point precision
    torch.set_float32_matmul_precision('high')

    # compile code if requested (pytorch 2.0)
    if cfg.location.compile == 'disable':
        print("Compiling disabled")
        module = module
    elif cfg.location.compile == 'default':
        print("Compiling default")
        module = torch.compile(module)
    elif cfg.location.compile == 'reduce-overhead':
        print("Compiling reduce-overhead")
        module = torch.compile(module, mode="reduce-overhead")
    elif cfg.location.compile == 'max-autotune':
        print("Compiling max-autotune")
        module = torch.compile(module, mode="max-autotune")

    metric_checkpoint = ModelCheckpoint(dirpath=cfg.location.result_dir + "/checkpoints",
                                       filename=run_name + "_best",
                                       verbose=True, monitor="Val Loss", mode="min")
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    progress_bar = TQDMProgressBar(refresh_rate=int(256/cfg.data.batch_size))
    callbacks = [lr_monitor,  progress_bar, metric_checkpoint] # more callbacks can be added

    trainer = pl.Trainer(max_epochs=cfg.num_epochs,
                         callbacks=callbacks, logger=logger,
                         accelerator='gpu', devices=cfg.location.n_gpus,
                         strategy=DDPStrategy(find_unused_parameters=False, process_group_backend=cfg.location.backend, timeout=timedelta(seconds=7200*4)),
                         accumulate_grad_batches=4, num_sanity_val_steps=0)

    trainer.fit(module, data_module)

    module._test_postfix = "Last"
    trainer.test(module, data_module.test_dataloader())
    module._test_postfix = "Best"
    trainer.test(module, data_module.test_dataloader(), ckpt_path="best")



if __name__ == "__main__":
    main()