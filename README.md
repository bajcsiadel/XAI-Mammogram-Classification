# Interpretable ResNet-50 based models for digital mammogram classification

## Setup

1. Install [Poetry](https://python-poetry.org/docs/)
2. Install environment
    ```shell
    poetry install
    ```
3. Activate environment
    ```shell
    poetry shell
    ```
4. Run the code
    ```shell
    python xai_mam/main.py 
    ```
   
## Data
The experiments included two datasets: the Mammographic Image Analysis Society (MIAS) [^Suckling-1994][^MIAS-link] database and the Digital Database for Screening Mammography (DDSM)[^Heath-1998][^Heath-2001][^DDSM-link] database.

## Included models

### ResNet

ResNet, introduced in [^He-2015], aims to address the problem of vanishing gradient in very deep neural networks by introducing residual connections between its blocks. In the current project ResNet-18 and ResNet-50 is included.

### ProtoPNet

The Prototypical Part Net [^Chen-2018], abbreviated as ProtoPNet, aims to learn patches of the image important for a defined classification task. We can choose any CNN model in order to extract the features from the patches. In this project we use ProtoPNet based on ResNet-18 and ResNet-50. 

### BagNet

BagNet [^Brendel-2019] is a modified ResNet-50 model adapting the bag-of-words model for image classification.

## Configuration

In this project `hydra` is used to configure the scrips. The configuration files are located in `xai_mam/conf`. The name of the configuration directory can be changed in `.env` by setting the `CONFIG_DIR_NAME` variable.

### Data

The available data configurations are available in `xai_mam/conf/data/set`. We have two mammogram datasets in this project:
* MIAS
* DDSM

Each dataset has four states (`xai_mam/conf/data/set/state`):
* original
* preprocessed
* masked
* masked_preprocessed

The preprocessing and masking (defining the breast) of the images was performed based on [^Bajcsi-2021].

Each dataset has two sizes (`xai_mam/conf/data/set/target/size):
* full: used for `normal vs abnormal` or `benign vs malignant` or `normal vs benign vs malignant` classification
* cropped: used for `benign vs malignant` classification only

This is automatically set when selecting a specific classification. For `benign vs malignant` classification the default is `cropped` but this can be changed.

By default, all images used. `xai_mam/conf/data/filters` contains filters e.g. only MLO.

The configuration contains predefined augmentations (`xai_mam/conf/data/augmentation`).

### Model

There are two models configured in this project (`xai_mam/conf/model`):
* ProtoPNet
* BagNet

The possible backbones (`xai_mam/conf/model/network`):
* ProtoPNet
  * ResNet-18
  * ResNet-50 
* BagNet
  * BagNet-9
  * BagNet-17
  * BagNet-33

The different train phases are set in `xai_mam/conf/model/phases`.

## Scripts

### Train a model

The main configuration file for the training is `xai_mam/conf/main_config.yaml`. 

The default parameters are the follow:
```
data/set=MIAS
data/set/target=normal_vs_abnormal
data/set/state=original
cross_validation=stratified
model=protopnet
model.params.prototypes.per_class=10
model.params.prototypes.size=256
model.params.prototypes.activation_fn=log
model/network=resnet18
model.network.add_on_layer_properties.type=bottleneck
model.network.add_on_layer_properties.activation=A
```

#### Data augmentation

To set augmentation to the training/validation set:
```
data/augmentation@data.set.image_properties.augmentations.train: <name>
data/augmentation@data.set.image_properties.augmentations.validation: <name>
```
`<name>` must be a valid configuration name in `xai_mam/conf/data/augmentation`.

#### Train only backbone

To train only the backbone set:
```
model/backbone_only@model=yes
```
---
For more details about setting configuration visit [Hydra](https://hydra.cc/docs/intro/).

## References

[^Bajcsi-2021]: Bajcsi A, Andreica A, Chira C. Towards feature selection for digital mammogram classification. Procedia Computer Science. 2021;192:632-41. [doi:10.1016/j.procs.2021.08.065](https://doi.org/10.1016/j.procs.2021.08.065)

[^Brendel-2019]: Brendel W, Bethge M. Approximating CNNs with Bag-of-local-Features models works surprisingly well on ImageNet. arXiv. Published online 3 2019. [doi:10.48550/arXiv.1904.00760](https://doi.org/10.48550/arXiv.1904.00760)

[^Chen-2018]: Chen C, Li O, Tao D, Barnett A, Rudin C, Su JK. This looks like that: deep learning for interpretable image recognition. Advances in neural information processing systems. 6 2018;32. [doi:10.48550/arXiv.1806.10574](https://doi.org/10.48550/arXiv.1806.10574)

[^He-2015]: He K, Zhang X, Ren S, and Sun J. Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2015. [doi:10.1109/CVPR.2016.90](https://doi.org/10.1109/CVPR.2016.90)

[^Heath-1998]: Heath M, Bowyer K, Kopans D, et al. Current Status of the Digital Database for Screening Mammography. In: Digital Mammography: Nijmegen, 1998. Springer Netherlands; 1998:457-460. [doi:10.1007/978-94-011-5318-8_75](https://doi.org/10.1007/978-94-011-5318-8_75)

[^Heath-2001]: Heath M, Bowyer K, Kopans D, Moore R, Kegelmeyer P. The digital database for screening mammography. In: Yaffe MJ, ed. Proceedings of the Fifth International Workshop on Digital Mammography. Medical Physics Publishing; 2001:212-218.

[^Suckling-1994]: Suckling J, Parker J, Dance DR. The mammographic image analysis society digital mammogram database. In: International Congress Series. Vol 1069. ; 01 1994:375-378.

[^MIAS-link]: University of Cambridge, School of Clinical Medicine. [Mammographic Image Analysis Society (MIAS)](https://www.repository.cam.ac.uk/items/b6a97f0c-3b9b-40ad-8f18-3d121eef1459) database v1.21.

[^DDSM-link]: University of South Florida. [DDSM: Digital Database for Screening Mammography](http://www.eng.usf.edu/cvprg/Mammography/Database.html)
