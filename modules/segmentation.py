import torch
import torchmetrics
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch.nn.functional as F
import segmentation_models_pytorch as smp

from torchmetrics.classification import MulticlassConfusionMatrix

from modules.utils import DiceLoss, calc_iou_scores, plot_confusion_matrix_asym



class Segmentation(pl.LightningModule):
    def __init__(self, cfg, wandb_id=""):
        super().__init__()
        # save config
        self._cfg = cfg

        # learning parameters
        self._lr = cfg.lr

        # general parameters
        self._classes = cfg.data.classes
        self._num_classes = cfg.data.num_classes

        self._overlap_train = cfg.data.overlap_train
        self._overlap_val = cfg.data.overlap_val

        # save all parameters
        self.save_hyperparameters()

        # to restore the wandb run
        self._wandb_id = wandb_id

        # for testing multiple times
        self._test_postfix = ""

        # create model
        self._model = smp.Unet(encoder_name="mit_b2", encoder_weights="imagenet", classes=2)
        self._final_act = torch.nn.Softmax(dim=1)

        # create dice_ce loss
        dice = DiceLoss()
        ce = torch.nn.CrossEntropyLoss()
        self._loss_f = lambda out, gt: cfg.ce_ratio*ce(out, gt) + cfg.dice_ratio*dice(self._final_act(out), gt)

        # confusion matrix
        self._conf_train = MulticlassConfusionMatrix(num_classes=self._num_classes)
        self._conf_val = MulticlassConfusionMatrix(num_classes=self._num_classes)
        self._conf_test = MulticlassConfusionMatrix(num_classes=self._num_classes)

        # for patient wise scores
        self._conf_val_inst = torch.nn.ModuleList([MulticlassConfusionMatrix(num_classes=self._num_classes) for i in range(cfg.data.num_val_inst)])
        self._conf_test_inst = torch.nn.ModuleList([MulticlassConfusionMatrix(num_classes=self._num_classes) for i in range(cfg.data.num_test_inst)])

        # loss accumulation
        self._loss_train = torchmetrics.MeanMetric()
        self._loss_val = torchmetrics.MeanMetric()


    def create_gt(self, labels):
        seg_labels = (labels>0).to(labels.dtype)
        seg_oh = F.one_hot(seg_labels.to(torch.long), num_classes=2).permute(0,3,1,2).to(torch.float32)
        return labels, seg_labels, seg_oh


    def forward(self, x, train=False):
        logits = self._model(x) 
        # process outputs
        probs = self._final_act(logits)
        seg = torch.argmax(probs, dim=1)
        return logits, probs, seg
    

    def training_step(self, batch, batch_idx):
        # run model on input images
        logits, probs, seg = self.forward(batch[0], train=True)

        # create gts
        labels, seg_labels, seg_oh = self.create_gt(batch[2])

        # calculate and accumulate loss
        o = self._overlap_train
        loss = self._loss_f(logits[:,:,o:-o,o:-o], seg_oh[:,:,o:-o,o:-o])
        self._loss_train.update(loss)

        # update confusion matrix
        self._conf_train.update(seg[:,o:-o,o:-o], labels[:,o:-o,o:-o])

        return {"loss": loss}


    def validation_step(self, batch, batch_idx):
        # run model on input images
        logits, probs, seg = self.forward(batch[0])

        # create gts
        labels, seg_labels, seg_oh = self.create_gt(batch[2])

        # calculate and accumulate loss
        o = self._overlap_val
        loss = self._loss_f(logits[:,:,o:-o,o:-o], seg_oh[:,:,o:-o,o:-o])
        self._loss_val.update(loss)

        # update confusion matrix, consider roi borders
        for i in range(len(batch[0])):
            c_x = int(batch[3][0][i])
            c_y = int(batch[3][1][i])
            wsi_nr = int(batch[4][i])
            self._conf_val.update(seg[None, i,o:-(o+c_y),o:-(o+c_x)], labels[None, i,o:-(o+c_y),o:-(o+c_x)])
            self._conf_val_inst[wsi_nr].update(seg[None, i,o:-(o+c_y),o:-(o+c_x)], labels[None, i,o:-(o+c_y),o:-(o+c_x)])

        return {"loss": loss}


    def test_step(self, batch, batch_idx):
        # run model on input images
        logits, probs, seg = self.forward(batch[0])

        # create gts
        labels = batch[2]

        # update confusion matrix, consider roi borders
        o = self._overlap_val
        for i in range(len(batch[0])):
            c_x = int(batch[3][0][i])
            c_y = int(batch[3][1][i])
            wsi_nr = int(batch[4][i])
            self._conf_test.update(seg[None, i,o:-(o+c_y),o:-(o+c_x)], labels[None, i,o:-(o+c_y),o:-(o+c_x)])
            self._conf_test_inst[wsi_nr].update(seg[None, i,o:-(o+c_y),o:-(o+c_x)], labels[None, i,o:-(o+c_y),o:-(o+c_x)])


    def on_train_epoch_end(self):
        # log training loss
        self.log("Train Loss", self._loss_train.compute())
        self._loss_train.reset()

        # calc confusion matrix
        conf_matrix = self._conf_train.compute().cpu().numpy()
        conf_matrix_norm = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)
        self._conf_train.reset()

        # adjust for combined tumor class
        conf_comb = np.copy(conf_matrix)
        conf_comb[1] = np.sum(conf_comb[1:], axis=0)
        conf_comb = conf_comb[:2,:2]

        # log dice scores
        ious = calc_iou_scores(conf_comb)
        self.log("Train IoU Score", ious[1], sync_dist=False)

        # log subtype variance
        subtype_var = np.var(conf_matrix_norm[1:-1,1])
        self.log("Train Subtype Var", subtype_var, sync_dist=False)

        # log confusion matrix
        cm_img = plot_confusion_matrix_asym(conf_matrix_norm[:,:2], ["BG", "Tumor"], self._classes)
        self.logger.log_image("Train Conf Matrix", images=[cm_img])

        # clear figures
        plt.close('all')
        plt.cla()
        plt.clf()


    def on_validation_epoch_end(self):
        # log validation loss
        self.log("Val Loss", self._loss_val.compute())
        self._loss_val.reset()

        # save wandb id
        if self._wandb_id == "":
            self._wandb_id = self.logger.version
            self.hparams["wandb_id"] = self._wandb_id

        # calc confusion matrix
        conf_matrix = self._conf_val.compute().cpu().numpy()
        conf_matrix_norm = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)
        self._conf_val.reset()

        # adjust for combined tumor class
        conf_comb = np.copy(conf_matrix)
        conf_comb[1] = np.sum(conf_comb[1:], axis=0)
        conf_comb = conf_comb[:2,:2]

        # log dice scores
        ious = calc_iou_scores(conf_comb)
        self.log("Val IoU Score", ious[1], sync_dist=False)

        # log subtype variance
        subtype_var = np.var(conf_matrix_norm[1:-1,1])
        self.log("Val Subtype Var", subtype_var, sync_dist=False)

        # log confusion matrix
        cm_img = plot_confusion_matrix_asym(conf_matrix_norm[:,:2], ["BG", "Tumor"], self._classes)
        self.logger.log_image("Val Conf Matrix", images=[cm_img])

        # log patient wise metrics
        patient_ious = []
        patient_confs = []
        for num, pat_conf in enumerate(self._conf_val_inst):
            # get patient confuson matrix
            conf_matrix = pat_conf.compute().cpu().numpy()
            pat_conf.reset()

            # store conf matrix
            patient_confs.append((num, conf_matrix))

            # adjust for combined tumor class
            conf_comb = np.copy(conf_matrix)
            conf_comb[1] = np.sum(conf_comb[1:], axis=0)
            conf_comb = conf_comb[:2,:2]

            patient_ious.append(calc_iou_scores(conf_comb)[1])

        self.log("Val Patient IoU Score", np.mean(np.array(patient_ious)))
        self.log("Val Patient IoU Var", np.var(np.array(patient_ious)))

        # task specific metrics
        if self._cfg.data.eval_subtypes:
            image_list = self.trainer.datamodule._data_modules[self._cfg.data.eval_key].val_dataset()._img_l

            self.log_subtype_iou("Val", image_list, patient_confs)

        # clear figures
        plt.close('all')
        plt.cla()
        plt.clf()


    def on_test_epoch_end(self):
        # calc confusion matrix
        conf_matrix = self._conf_test.compute().cpu().numpy()
        conf_matrix_norm = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)
        self._conf_test.reset()

        # adjust for combined tumor class
        conf_comb = np.copy(conf_matrix)
        conf_comb[1] = np.sum(conf_comb[1:], axis=0)
        conf_comb = conf_comb[:2,:2]

        # log dice scores
        ious = calc_iou_scores(conf_comb)
        self.log(f"Test IoU Score {self._test_postfix}", ious[1], sync_dist=False)

        # log subtype variance
        subtype_var = np.var(conf_matrix_norm[1:-1,1])
        self.log(f"Test Subtype Var {self._test_postfix}", subtype_var, sync_dist=False)

        # log confusion matrix
        cm_img = plot_confusion_matrix_asym(conf_matrix_norm[:,:2], ["BG", "Tumor"], self._classes)
        self.logger.log_image(f"Test Conf Matrix {self._test_postfix}", images=[cm_img])

        # log patient wise metrics
        patient_ious = []
        patient_confs = []
        for num, pat_conf in enumerate(self._conf_test_inst):
            # get patient confuson matrix
            conf_matrix = pat_conf.compute().cpu().numpy()
            pat_conf.reset()

            # store conf matrix
            patient_confs.append((num, conf_matrix))

            # adjust for combined tumor class
            conf_comb = np.copy(conf_matrix)
            conf_comb[1] = np.sum(conf_comb[1:], axis=0)
            conf_comb = conf_comb[:2,:2]

            patient_ious.append(calc_iou_scores(conf_comb)[1])

        self.log(f"Test Patient IoU Score {self._test_postfix}", np.mean(np.array(patient_ious)))
        self.log(f"Test Patient IoU Var {self._test_postfix}", np.var(np.array(patient_ious)))
        img_name = list(self.trainer.datamodule._data_modules[self._cfg.data.eval_key].test_dataset().wsi_lookup.keys())
        img_name = [str(el) for el in img_name]
        self.logger.log_table(f"Test Patient IoUs {self._test_postfix}", columns=img_name, data=[patient_ious])

        # task specific metrics
        if self._cfg.data.eval_subtypes:
            image_list = self.trainer.datamodule._data_modules[self._cfg.data.eval_key].test_dataset()._img_l

            self.log_subtype_iou("Test", image_list, patient_confs, self._test_postfix)

        # clear figures
        plt.close('all')
        plt.cla()
        plt.clf()


    def log_subtype_iou(self, mode, image_list, patient_confs, post_fix = ""):
        tumor_subtypes = self._classes[1:]
        subtype_lists = []
        [subtype_lists.append([]) for subtype in tumor_subtypes]

        for conf in patient_confs:
            image_tup = image_list[conf[0]]

            for num, tumor_subtype in enumerate(tumor_subtypes):
                if tumor_subtype in image_tup[self._cfg.data.name_idx]:
                    subtype_lists[num].append(conf[1])
        
        # sum conf matrix of each subtype, calc IOU
        subtype_iou = []
        subtype_imgs = []
        present_subtypes = []
        for subtype_list, tumor_subtype in zip(subtype_lists,tumor_subtypes):
            if len(subtype_list) > 0:
                conf_sum = np.zeros_like(subtype_list[0])
                for subtype_conf in subtype_list:
                    conf_sum += subtype_conf

                # combine for combined tumor class
                conf_sum[1] = np.sum(conf_sum[1:], axis=0)
                conf_sum = conf_sum[:2,:2]

                ious = calc_iou_scores(conf_sum)
                subtype_iou.append(ious[1])

                conf_norm = conf_sum / conf_sum.sum(axis=1, keepdims=True)
                cm_img = plot_confusion_matrix_asym(conf_norm, ["BG", "Tumor"], ["BG", "Tumor"], tumor_subtype)
                subtype_imgs.append(cm_img)
                present_subtypes.append(tumor_subtype)

        # log subtype confs
        self.logger.log_image(f"{mode} Subtype Confs {self._test_postfix}", images=subtype_imgs)

        # log IoUs
        self.logger.log_table(f"{mode} Subtype IoUs {post_fix}", columns=present_subtypes, data=[subtype_iou])

        # calc and log Subtype IoU Variance
        self.log(f"{mode} Subtype IoU Var {post_fix}", np.var(np.array(subtype_iou)), sync_dist=False)
        self.log(f"{mode} Subtype IoU Score {post_fix}", np.mean(np.array(subtype_iou)), sync_dist=False)


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self._model.parameters(), lr=self._lr)
        return  optimizer