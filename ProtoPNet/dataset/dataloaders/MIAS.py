import gin

import ProtoPNet.dataset.dataloaders as dl

from ProtoPNet.dataset.metadata import DATASETS


class MIASDataset(dl.CustomVisionDataset):
    def __init__(self, classification, subset="train", normalize=True):
        super().__init__(DATASETS["MIAS"], classification, subset=subset, normalize=normalize)


@gin.configurable
class MIASDataModule(dl.CustomDataModule):
    """
    DataModule to define data loaders for MIAS dataset

    :param used_images:
    :type used_images: str
    :param classification:
    :type classification: str
    :param cross_validation_folds: Number of cross validation folds
    :type cross_validation_folds: int
    :param stratified:
    :type stratified: bool
    :param groups:
    :type groups: bool
    :param num_workers:
    :type num_workers: int
    :param seed:
    :type seed: int
    """
    def __init__(
            self,
            used_images,
            classification,
            cross_validation_folds=5,
            stratified=True,
            groups=True,
            num_workers=4,
            seed=0,
    ):
        super().__init__(
            DATASETS["MIAS"],
            used_images,
            classification,
            cross_validation_folds,
            stratified,
            groups,
            num_workers,
            seed,
            dataset_class=MIASDataset
        )
