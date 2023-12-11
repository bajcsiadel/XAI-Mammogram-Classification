import albumentations as A
import gin

import ProtoPNet.dataset.dataloaders as dl

from ProtoPNet.dataset.metadata import DATASETS


class MIASDataset(dl.CustomVisionDataset):
    def __init__(self, classification, subset="train", data_filters=None, normalize=True, transform=A.NoOp()):
        super().__init__(
            DATASETS["MIAS"],
            classification,
            subset=subset,
            data_filters=data_filters,
            normalize=normalize,
            transform=transform
        )


@gin.configurable
class MIASDataModule(dl.CustomDataModule):
    """
    DataModule to define data loaders for MIAS dataset

    :param used_images:
    :type used_images: str
    :param classification:
    :type classification: str
    :param data_filters: Filters to apply to the data
    :type data_filters: typ.List[typ.Callable[[pd.DataFrame], pd.DataFrame]] | None
    :param cross_validation_folds: Number of cross validation folds
    :type cross_validation_folds: int
    :param stratified:
    :type stratified: bool
    :param balanced:
    :type balanced: bool
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
            data_filters=None,
            cross_validation_folds=5,
            stratified=True,
            balanced=False,
            groups=True,
            num_workers=4,
            seed=0,
    ):
        super().__init__(
            DATASETS["MIAS"],
            used_images,
            classification,
            data_filters,
            cross_validation_folds,
            stratified,
            balanced,
            groups,
            num_workers,
            seed,
            dataset_class=MIASDataset
        )
