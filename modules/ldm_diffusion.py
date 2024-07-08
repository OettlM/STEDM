import torch
import torchmetrics
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl

from PIL import Image
from omegaconf import OmegaConf

from networks.s_zss_dm import S_ZSS_DM



class LDM_Diffusion(pl.LightningModule):
    def __init__(self, cfg, wandb_id=""):
        super().__init__()
        # save config
        self._cfg = cfg

        # learning parameters
        self._lr = cfg.lr

        # save all parameters
        self.save_hyperparameters()

        # to restore the wandb run
        self._wandb_id = wandb_id

        # prepare ldm
        ldm_conf = cfg.diffusion
        # set ae path correctly
        ldm_conf.first_stage_config.params.ckpt_path = cfg.location.result_dir + "/" + ldm_conf.first_stage_config.params.ckpt_path
        # set pretrained unet path if exists
        if hasattr(ldm_conf, "ckpt_path"):
            ldm_conf.ckpt_path = cfg.location.result_dir + "/" + ldm_conf.ckpt_path

        ldm_dict = OmegaConf.to_container(ldm_conf)
        self._model = S_ZSS_DM(encoder="swin_v2_t", sampling_cfg=cfg.style_sampling, agg_cfg=cfg.style_agg, cfg=cfg, **ldm_dict)

        # register ldm
        self.register_module("model", self._model)

        # loss accumulation
        self._train_loss = torchmetrics.MeanMetric()


    def forward(self, x, *args, **kwargs):
        self._model.forward(x, *args, **kwargs)


    def prepare_batch(self, batch):
        img = batch[0].permute(0,2,3,1)
        seg_oh = batch[1].permute(0,2,3,1)
        style = batch[3].permute(0,1,3,4,2)

        seg_oh[:,:,:,1] = torch.sum(seg_oh[:,:,:,1:], dim=-1)
        seg_oh = seg_oh[:,:,:,:2]

        ldm_batch = {"image": img, "segmentation": seg_oh, "style_imgs":style}
        return ldm_batch


    def training_step(self, batch, batch_idx):
        # prepare batch for ldm
        ldm_batch = self.prepare_batch(batch)

        # run batch and calculate loss
        loss = self._model.training_step(ldm_batch, batch_idx)

        # accumulate loss
        self._train_loss.update(loss)

        return loss


    def predict_step(self, batch, batch_idx):
        ldm_batch = self.prepare_batch(batch)

        z, c_0 = self._model.get_input(ldm_batch, "image")

        if (self._cfg.cfg_scale == 1) or (self._cfg.style_sampling.name == "none"):
            out_imgs, _ = self._model.sample_log(c_0, batch_size=len(z), ddim=True, ddim_steps=self._cfg.ddim_steps, eta=self._cfg.eta, log_every_t=1000)
            torch.cuda.empty_cache()
        else:
            # get uncond conditioning
            ldm_batch_uncond = {"image": torch.zeros_like(ldm_batch["image"]), "segmentation": ldm_batch["segmentation"], "style_imgs":torch.zeros_like(ldm_batch["style_imgs"])-2}
            z, c_uncond = self._model.get_input(ldm_batch_uncond, "image")

            # sample new images
            out_imgs, _ = self._model.sample_log(c_0, batch_size=len(z), ddim=True, ddim_steps=self._cfg.ddim_steps, eta=self._cfg.eta, log_every_t=1000, unconditional_conditioning=c_uncond, unconditional_guidance_scale=self._cfg.cfg_scale)
            torch.cuda.empty_cache()

        # prepare images for saving
        out_imgs = torch.clip(self._model.decode_first_stage(out_imgs), -1, 1)
        torch.cuda.empty_cache()
        out_imgs = ((out_imgs.permute(0,2,3,1).cpu().numpy() + 1) * 127.5).astype(np.uint8)

        # save images
        for img, seg, num in zip(out_imgs, torch.argmax(ldm_batch["segmentation"], dim=-1).cpu().numpy().astype(np.uint8), batch[4].cpu().numpy()):
            out_path = self.predict_dir

            # pad number to 5 digits
            num_str = str(num).zfill(5)

            # save img and seg
            Image.fromarray(img).save(out_path + f"/img_{num_str}.png")
            Image.fromarray(seg).save(out_path + f"/seg_{num_str}.png")


    def on_train_batch_start(self, batch, batch_idx):
        self._model.on_train_batch_start(batch, batch_idx, -1)


    def on_train_batch_end(self, *args, **kwargs):
        self._model.on_train_batch_end(*args, **kwargs)


    def on_train_epoch_end(self):
        self.log("Train Loss", self._train_loss.compute())
        self._train_loss.reset()

        # save wandb id
        if self._wandb_id == "":
            self._wandb_id = self.logger.version
            self.hparams["wandb_id"] = self._wandb_id

        # create example images
        if hasattr(self._cfg.data, "test_folder"):         
            # test image folder path
            test_folder_path = self._cfg.location.data_dir + "/" + self._cfg.data.test_folder

            # load test condition image
            test_img = np.array(Image.open(test_folder_path + "/test_c.png").convert('L'))
            test_img = (test_img > 0).astype(np.uint8)
            out_ch = 2

            c = F.one_hot(torch.from_numpy(test_img).to(self.device).to(torch.long), num_classes=out_ch).unsqueeze(0).to(torch.float32)

            # load style images
            test_style_path = test_folder_path + "/" + self._cfg.style_sampling.name

            with torch.no_grad():
                # set to validation
                self._model.eval()

                if self._cfg.style_sampling.name == "nearby":
                    style_0 = (torch.from_numpy(np.array(Image.open(test_style_path + "/0_img.png"))[:,:,:3]).to(torch.float32).to(self.device).unsqueeze(0).unsqueeze(0) / 127.5)-1
                    style_1 = (torch.from_numpy(np.array(Image.open(test_style_path + "/1_img.png"))[:,:,:3]).to(torch.float32).to(self.device).unsqueeze(0).unsqueeze(0) / 127.5)-1
                    style_2 = (torch.from_numpy(np.array(Image.open(test_style_path + "/2_img.png"))[:,:,:3]).to(torch.float32).to(self.device).unsqueeze(0).unsqueeze(0) / 127.5)-1
                    style_3 = (torch.from_numpy(np.array(Image.open(test_style_path + "/3_img.png"))[:,:,:3]).to(torch.float32).to(self.device).unsqueeze(0).unsqueeze(0) / 127.5)-1
                elif self._cfg.style_sampling.name == "mp":
                    style_0 = []
                    style_1 = []
                    style_2 = []
                    style_3 = []
                    for i in range(self._cfg.style_sampling.num_patches):
                        style_0.append((torch.from_numpy(np.array(Image.open(test_style_path + f"/0_img_{i}.png"))[:,:,:3]).to(torch.float32).to(self.device).unsqueeze(0).unsqueeze(0) / 127.5)-1)
                        style_1.append((torch.from_numpy(np.array(Image.open(test_style_path + f"/1_img_{i}.png"))[:,:,:3]).to(torch.float32).to(self.device).unsqueeze(0).unsqueeze(0) / 127.5)-1)
                        style_2.append((torch.from_numpy(np.array(Image.open(test_style_path + f"/2_img_{i}.png"))[:,:,:3]).to(torch.float32).to(self.device).unsqueeze(0).unsqueeze(0) / 127.5)-1)
                        style_3.append((torch.from_numpy(np.array(Image.open(test_style_path + f"/3_img_{i}.png"))[:,:,:3]).to(torch.float32).to(self.device).unsqueeze(0).unsqueeze(0) / 127.5)-1)
                    style_0 = torch.concat(style_0, dim=1)
                    style_1 = torch.concat(style_1, dim=1)
                    style_2 = torch.concat(style_2, dim=1)
                    style_3 = torch.concat(style_3, dim=1)
                elif self._cfg.style_sampling.name == "dummy":
                    style_0 = (torch.zeros((1,1,512,512,3), dtype=torch.float32, device=self.device) / 127.5)-1
                    style_1 = style_0
                    style_2 = style_0
                    style_3 = style_0

                ldm_batch_0 = {"image": torch.zeros((1,512,512,3), dtype=torch.float32, device=self.device), "segmentation": c, "style_imgs":style_0}
                ldm_batch_1 = {"image": torch.zeros((1,512,512,3), dtype=torch.float32, device=self.device), "segmentation": c, "style_imgs":style_1}
                ldm_batch_2 = {"image": torch.zeros((1,512,512,3), dtype=torch.float32, device=self.device), "segmentation": c, "style_imgs":style_2}
                ldm_batch_3 = {"image": torch.zeros((1,512,512,3), dtype=torch.float32, device=self.device), "segmentation": c, "style_imgs":style_3}
            
                # create conditioning tensor
                z, c_0 = self._model.get_input(ldm_batch_0, "image")
                z, c_1 = self._model.get_input(ldm_batch_1, "image")
                z, c_2 = self._model.get_input(ldm_batch_2, "image")
                z, c_3 = self._model.get_input(ldm_batch_3, "image")

                # images
                img_ddim_w0, _ = self._model.sample_log(c_0, batch_size=1, ddim=True, ddim_steps=128, eta=0.0, log_every_t=200)
                img_ddim_w1, _ = self._model.sample_log(c_1, batch_size=1, ddim=True, ddim_steps=128, eta=0.0, log_every_t=200)
                img_ddim_w2, _ = self._model.sample_log(c_2, batch_size=1, ddim=True, ddim_steps=128, eta=0.0, log_every_t=200)
                img_ddim_w3, _ = self._model.sample_log(c_3, batch_size=1, ddim=True, ddim_steps=128, eta=0.0, log_every_t=200)
                
                img_ddim_w0 = torch.clip(self._model.decode_first_stage(img_ddim_w0), -1, 1)
                img_ddim_w1 = torch.clip(self._model.decode_first_stage(img_ddim_w1), -1, 1)
                img_ddim_w2 = torch.clip(self._model.decode_first_stage(img_ddim_w2), -1, 1)
                img_ddim_w3 = torch.clip(self._model.decode_first_stage(img_ddim_w3), -1, 1)

                if (self._cfg.style_drop_rate > 0.0) and (self._cfg.style_sampling.name != "dummy"):
                    ldm_batch_uncond = {"image": torch.zeros((1,512,512,3), dtype=torch.float32, device=self.device), "segmentation": c, "style_imgs":torch.zeros_like(style_0)-2}
                    z, c_uncond = self._model.get_input(ldm_batch_uncond, "image")

                    # images
                    img_ddim_w0_uncond, _ = self._model.sample_log(c_0, batch_size=1, ddim=True, ddim_steps=128, eta=0.0, log_every_t=200, unconditional_conditioning=c_uncond, unconditional_guidance_scale=3.0)
                    img_ddim_w1_uncond, _ = self._model.sample_log(c_0, batch_size=1, ddim=True, ddim_steps=128, eta=0.0, log_every_t=200, unconditional_conditioning=c_uncond, unconditional_guidance_scale=5.0)
                    img_ddim_w2_uncond, _ = self._model.sample_log(c_1, batch_size=1, ddim=True, ddim_steps=128, eta=0.0, log_every_t=200, unconditional_conditioning=c_uncond, unconditional_guidance_scale=3.0)
                    img_ddim_w3_uncond, _ = self._model.sample_log(c_1, batch_size=1, ddim=True, ddim_steps=128, eta=0.0, log_every_t=200, unconditional_conditioning=c_uncond, unconditional_guidance_scale=5.0)
                    
                    img_ddim_w0_uncond = torch.clip(self._model.decode_first_stage(img_ddim_w0_uncond), -1, 1)
                    img_ddim_w1_uncond = torch.clip(self._model.decode_first_stage(img_ddim_w1_uncond), -1, 1)
                    img_ddim_w2_uncond = torch.clip(self._model.decode_first_stage(img_ddim_w2_uncond), -1, 1)
                    img_ddim_w3_uncond = torch.clip(self._model.decode_first_stage(img_ddim_w3_uncond), -1, 1)

                    img_ddim_w0_uncond = ((img_ddim_w0_uncond[0].permute(1,2,0).cpu().numpy() + 1) * 127.5).astype(np.uint8)
                    img_ddim_w1_uncond = ((img_ddim_w1_uncond[0].permute(1,2,0).cpu().numpy() + 1) * 127.5).astype(np.uint8)
                    img_ddim_w2_uncond = ((img_ddim_w2_uncond[0].permute(1,2,0).cpu().numpy() + 1) * 127.5).astype(np.uint8)
                    img_ddim_w3_uncond = ((img_ddim_w3_uncond[0].permute(1,2,0).cpu().numpy() + 1) * 127.5).astype(np.uint8)

                    self.logger.log_image("Sample Images CFG", images=[img_ddim_w0_uncond, img_ddim_w1_uncond, img_ddim_w2_uncond, img_ddim_w3_uncond], caption=["Test 0", "Test 1", "Test 2", "Test 3"])

            img_ddim_w0 = ((img_ddim_w0[0].permute(1,2,0).cpu().numpy() + 1) * 127.5).astype(np.uint8)
            img_ddim_w1 = ((img_ddim_w1[0].permute(1,2,0).cpu().numpy() + 1) * 127.5).astype(np.uint8)
            img_ddim_w2 = ((img_ddim_w2[0].permute(1,2,0).cpu().numpy() + 1) * 127.5).astype(np.uint8)
            img_ddim_w3 = ((img_ddim_w3[0].permute(1,2,0).cpu().numpy() + 1) * 127.5).astype(np.uint8)

            # log images to wandb
            self.logger.log_image("Sample Images", images=[img_ddim_w0, img_ddim_w1, img_ddim_w2, img_ddim_w3], caption=["Test 0", "Test 1", "Test 2", "Test 3"])


    def configure_optimizers(self):
        params = list(self.model.model.parameters())
        if self.model.cond_stage_trainable:
            params = params + list(self.model.cond_stage_model.parameters())
        if self.model.learn_logvar:
            params.append(self.model.logvar)
        if hasattr(self.model, "embedder"):
            params = params + list(self.model.embedder.parameters())

        optimizer = torch.optim.AdamW(params, lr=self._lr)
        return optimizer