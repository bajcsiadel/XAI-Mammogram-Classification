import gin

import ProtoPNet.dataset.dataloaders as dl

from ProtoPNet.dataset.metadata import DATASETS


class DDSMDataset(dl.CustomVisionDataset):
    def __init__(self, classification, subset="train"):
        super().__init__(DATASETS["DDSM"], classification, subset=subset)


@gin.configurable
class DDSMDataModule(dl.CustomDataModule):
    """
    DataModule to define data loaders for DDSM dataset.

    :param used_images:
    :type used_images: str
    :param classification:
    :type classification: str
    :param batch_size: Number of samples in a batch
    :type batch_size: int
    :param cross_validation_folds: Number of cross validation folds
    :type cross_validation_folds: int
    :param stratified:
    :type stratified: bool
    :param groups:
    :type groups: bool
    :param push_batch_size:
    :type push_batch_size: int
    :param num_workers:
    :type num_workers: int
    :param seed:
    :type seed: int
    """
    def __init__(
            self,
            used_images,
            classification,
            batch_size,
            cross_validation_folds,
            stratified,
            groups,
            push_batch_size,
            num_workers,
            seed
    ):
        super().__init__(
            DATASETS["DDSM"],
            used_images,
            classification,
            batch_size,
            cross_validation_folds,
            stratified,
            groups,
            push_batch_size,
            num_workers,
            seed,
            dataset_class=DDSMDataset
        )
