# Style-Extracting Diffusion Models for Semi-Supervised Histopathology Segmentation

### ECCV 2024

[arXiv](https://arxiv.org/abs/2403.14429)

<p align="center">
<img src=assets/overview.png />
</p>

<p align="center">
<img src=assets/her2_examples.png />
</p>


## Paper Abstract

Deep learning-based image generation has seen significant advancements with diffusion models, notably improving the quality of generated images. Despite these developments, generating images with unseen characteristics beneficial for downstream tasks has received limited attention. To bridge this gap, we propose Style-Extracting Diffusion Models, featuring two conditioning mechanisms. 
Specifically, we utilize 1) a style conditioning mechanism which allows to inject style information of previously unseen images during image generation and 2) a content conditioning which can be targeted to a downstream task, e.g., layout for segmentation.
We introduce a trainable style encoder to extract style information from images, and an aggregation block that merges style information from multiple style inputs. This architecture enables the generation of images with unseen styles in a zero-shot manner, by leveraging styles from unseen images, resulting in more diverse generations. In this work, we use the image layout as target condition and first show the capability of our method on a natural image dataset as a proof-of-concept. We further demonstrate its versatility in histopathology, where we combine prior knowledge about tissue composition and unannotated data to create diverse synthetic images with known layouts. This allows us to generate additional synthetic data to train a segmentation network in a semi-supervised fashion. We verify the added value of the generated images by showing improved segmentation results and lower performance variability between patients when synthetic images are included during segmentation training. 


## Requirements
A suitable [conda](https://conda.io/) environment named `STEDM` can be created
and activated with:

```
conda env create -f environment.yaml
conda activate STEDM
```

## Usage

#### Training the diffusion model

```
python train_diff.py location=cluster data=flowers style_sampling=augmented style_agg=mean data.class_train_samples=3
```

#### Generating synthetic images

```
python predict_diff.py location=cluster data=flowers data/dataset=[flowers_anno,flowers_unanno] data.ratios=[1.0,1.0] style_sampling=augmented style_agg=mean data.class_train_samples=3 cfg_scale=1.5 +ckpt_name=YOUR_CKPT_NAME.ckpt +predict_dir=RESULTS_SUB_FOLDER
```


#### Training the segmentation model

```
python train_seg.py location=cluster data=flowers data/dataset=[her2_anno,her2_syn] data.ratios=[0.2,0.8] data.dataset.flowers_syn.folder_name=RESULTS_SUB_FOLDER data.class_train_samples=3
```

#### Settings

Parameterization of runs is performed with hydra, with the configs being located at conf

**location**, contains settings for the execution locations, e.g. number of gpus

**data**, the dataset to be used

**data/datasets**, list of the sub-datasets used in the current experiment

**data/ratios**, list matching the len of data/datasets, with the sampling ratios for the datasets

**style_sampling**, the style sampling strategy used

**style_agg**, the style aggregation strategy used

**data.class_train_samples**, the number of samples used from each class. Controlls the amount of labelled data utilized. Unused data is used as unlabelled style source.

**cfg_scale**, classifier-free guidance scale, utilized for generating synthetic images


#### Data

The data structure of the flowers dataset is the same as the original one, from where the dataset was downloaded.
For the HER2 and CATCH dataset, the images and annotations were available via [EXACT](https://github.com/DeepMicroscopy/Exact).
We still provide the dataloaders, so the usage can be seen and new dataloaders can be adapted following a similar structure.