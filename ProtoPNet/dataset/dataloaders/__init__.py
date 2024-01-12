import copy
import typing as typ

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import pipe
import torch
from albumentations.pytorch import ToTensorV2
from hydra.utils import instantiate
from icecream import ic
import omegaconf
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets

from ProtoPNet.dataset.metadata import DataFilter
from ProtoPNet.util import config_types as conf_typ


def _target_transform(target):
    return torch.tensor(target, dtype=torch.long)


def my_collate_function(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]


class CustomVisionDataset(datasets.VisionDataset):
    """
    Custom Vision Dataset class for PyTorch.

    :param dataset_meta: Dataset metadata
    :type dataset_meta: conf_typ.Dataset
    :param classification: Classification type
    :type classification: str
    :param subset: Subset to use
    :type subset: str
    :param data_filters: Filters to apply to the data
    :type data_filters: typ.List[DataFilter] | None
    :param transform: Transform to apply to the images
    :type transform: albumentations.BasicTransform | list[albumentations.BasicTransform]
    :param target_transform: Transform to apply to the targets
    :type target_transform: callable
    """

    def __init__(
        self,
        dataset_meta,
        classification,
        subset="train",
        data_filters=None,
        normalize=True,
        transform=A.NoOp(),
        target_transform=_target_transform,
    ):
        if isinstance(transform, A.BasicTransform):
            transform = [transform]
        elif isinstance(transform, omegaconf.ListConfig):
            transform = [instantiate(t) for t in transform]

        if normalize:
            transform.append(
                A.Normalize(
                    mean=dataset_meta.image_properties.mean,
                    std=dataset_meta.image_properties.std,
                    max_pixel_value=dataset_meta.image_properties.max_value,
                )
            )
            dataset_meta.image_properties.max_value = 1.0

        transform = A.Compose(
            [
                A.ToFloat(max_value=dataset_meta.image_properties.max_value),
                A.Resize(
                    width=dataset_meta.image_properties.width,
                    height=dataset_meta.image_properties.height,
                ),
                *transform,  # unpack list of transforms
                ToTensorV2(),
            ]
        )

        super().__init__(
            str(dataset_meta.image_dir),
            transform=transform,
            target_transform=target_transform,
        )

        assert subset in [
            "train",
            "test",
            "all",
        ], "subset must be one of 'train', 'test' or 'all'"
        self.__subset = subset

        self.__dataset_meta = dataset_meta

        metafile_params = dataset_meta.metadata.parameters
        if isinstance(metafile_params, omegaconf.DictConfig):
            metafile_params = omegaconf.OmegaConf.to_object(dataset_meta.metadata.parameters)
        if isinstance(metafile_params, conf_typ.CSVParameters) or type(metafile_params).__name__ == "CSVParameters":
            metafile_params = metafile_params.to_dict()

        self.__meta_information = pd.read_csv(
            dataset_meta.metadata.file, **metafile_params
        )

        if data_filters is not None:
            for data_filter in data_filters:
                if isinstance(data_filter, omegaconf.DictConfig):
                    data_filter = omegaconf.OmegaConf.to_object(data_filter)
                self.__meta_information = data_filter(self.__meta_information)

            assert len(self.__meta_information) > 0, "no data left after filtering"

        assert isinstance(
            self.__meta_information.columns, pd.MultiIndex
        ), "metadata does not have split information"
        cls_types = list(
            self.__meta_information.columns.get_level_values(0).tolist()
            | pipe.where(lambda column: "_vs_" in column)
        )
        assert (
            len(cls_types) > 0
        ), f"No classification types found in the metadata {dataset_meta.metadata.file}"
        assert classification in cls_types, (
            f"cls_types, classification must be one from "
            f"{cls_types}, not {classification}"
        )
        self.__classification = classification

        self.__meta_information = self.__meta_information[
            self.__meta_information[(self.__classification, "label")].notna()
        ]
        # Filter out the rows that are not in the subset
        if subset != "all":
            self.__meta_information = self.__meta_information[
                self.__meta_information[(self.__classification, "subset")]
                == self.__subset
            ]
        self.__classes = (
            self.__meta_information[(self.__classification, "label")].unique().tolist()
        )
        self.__dataset_meta.number_of_classes = len(self.__classes)
        self.__class_to_number = {cls: i for i, cls in enumerate(self.__classes)}

        self.__transform = transform
        self.__target_transform = target_transform

        for transform in self.__transform[::-1]:
            if isinstance(transform, A.RandomResizedCrop) or isinstance(
                transform, A.Resize
            ):
                self.__dataset_meta.input_size = [transform.height, transform.width]
                break

    @property
    def targets(self):
        """
        Get the targets of the dataset

        :return:
        :rtype: np.ndarray
        """
        return (
            self.__meta_information[(self.__classification, "label")].to_numpy().copy()
        )

    @property
    def metadata(self):
        """
        Get metadata of the dataset

        :return:
        :rtype: pd.DataFrame
        """
        return self.__meta_information.copy()

    def __repr__(self):
        """
        Get the representation of the dataset

        :return:
        :rtype: str
        """
        formatted_transform = "\n\t".join(str(self.__transform).split("\n"))

        dataset_ = (
            self.__dataset_meta.name
            if self.__dataset_meta.name != ""
            else "CustomVisionDataset"
        )
        return (
            f"Dataset {dataset_}\n"
            f"\tSubset: {self.__subset}\n"
            f"\tNumber of datapoints: {len(self)}\n"
            f"\tImage location: {self.__dataset_meta.image_dir}\n"
            f"\tTransform: {formatted_transform}\n"
        )

    def __len__(self):
        """
        Get the number of images in the dataset

        :return: number of images
        :rtype: int
        """
        return len(self.__meta_information)

    def __getitem__(self, index):
        """
        Get sample at a specific index from the dataset

        :param index: Index of the sample
        :type index: int
        :return: Return the image and its label at a given index
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
        sample = self.__meta_information.iloc[index]

        # Load the image
        image_path = self.__dataset_meta.image_dir / (
            f"{sample.name[1]}" f"{self.__dataset_meta.image_properties.extension}"
        )

        if self.__dataset_meta.image_properties.extension in [".npy", ".npz"]:
            # quicker to load than cv2.imread
            image = np.load(image_path)["image"]
        else:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        target = self.__class_to_number[sample[self.__classification, "label"]]

        # apply transforms
        image = self.__transform(image=image)["image"]
        target = self.__target_transform(target)

        return image, target


class CustomDataModule:
    """
    DataModule to define data loaders.
    :param data:
    :type data: Dataset
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
        data,
        classification,
        data_filters=None,
        cross_validation_folds=None,
        stratified=False,
        balanced=False,
        grouped=False,
        num_workers=0,
        seed=None,
    ):
        self.__data = data

        dataset_params = {
            "dataset_meta": self.__data,
            "classification": classification,
            "data_filters": data_filters,
        }

        # define datasets
        self.__train_data = CustomVisionDataset(
            **dataset_params,
            transform=self.__data.image_properties.augmentations.train,
            subset="train",
        )
        self.__push_data = CustomVisionDataset(
            **dataset_params,
            transform=self.__data.image_properties.augmentations.push,
            normalize=False,
            subset="train",
        )
        self.__test_data = CustomVisionDataset(
            **dataset_params,
            transform=self.__data.image_properties.augmentations.test,
            subset="test",
        )

        self.__number_of_workers = num_workers

        self.__init_cross_validation(
            cross_validation_folds, stratified, balanced, groups, seed
        )

    def __init_cross_validation(
        self, cross_validation_folds, stratified, balanced, groups, seed
    ):
        """
        Initialize cross validation folds
        :param cross_validation_folds: number of cross validation folds
        :type cross_validation_folds: int
        :param stratified:
        :type stratified: bool
        :param balanced:
        :type balanced: bool
        :param groups:
        :type groups: bool
        :param seed:
        :type seed: int
        """
        if cross_validation_folds in [None, 0, 1]:
            self.__folds = {}
            return

        targets = self.__train_data.targets
        sample_groups = self.__train_data.metadata.index.get_level_values(
            "patient_id"
        ).to_numpy()

        cv_kwargs = {}
        if groups or balanced:
            cv_kwargs["groups"] = sample_groups
        if stratified or balanced:
            cv_kwargs["y"] = targets

        if balanced:
            from ProtoPNet.util.split_data.cross_validation import BalancedGroupKFold

            cross_validator_class = BalancedGroupKFold
        elif stratified and groups:
            from sklearn.model_selection import StratifiedGroupKFold

            cross_validator_class = StratifiedGroupKFold
        elif stratified and not groups:
            from sklearn.model_selection import StratifiedKFold

            cross_validator_class = StratifiedKFold
        elif not stratified and groups:
            from sklearn.model_selection import GroupKFold

            cross_validator_class = GroupKFold
        else:
            # not stratified and not groups
            from sklearn.model_selection import KFold

            cross_validator_class = KFold

        cross_validator = cross_validator_class(
            n_splits=cross_validation_folds, shuffle=True, random_state=seed
        )

        self.__folds = {
            fold
            + 1: (
                SubsetRandomSampler(train_idx),
                SubsetRandomSampler(validation_idx),
            )
            for fold, (train_idx, validation_idx) in enumerate(
                cross_validator.split(self.__train_data, **cv_kwargs)
            )
        }

    @property
    def dataset(self):
        return copy.deepcopy(self.__data)

    @property
    def folds(self):
        """
        Generate the folds and the corresponding samplers
        :return: fold number, (train sampler, validation sampler)
        :rtype: typ.Generator[int, typ.Tuple[SubsetRandomSampler, SubsetRandomSampler]]
        """
        yield from self.__folds.items()

    def __get_data_loader(self, dataset, **kwargs):
        """
        Get a data loader for a given dataset
        :param dataset:
        :type dataset: CustomVisionDataset
        :param kwargs:
        :type kwargs: typ.Dict[str, typ.Any]
        :return: data loader
        :rtype: DataLoader
        """
        return DataLoader(dataset, num_workers=self.__number_of_workers, **kwargs)

    def train_dataloader(self, batch_size, sampler=None, **kwargs):
        """
        Get a data loader for the training set
        :param batch_size:
        :type batch_size: int
        :param sampler:
        :type sampler: torch.utils.data.Sampler | typ.Iterable[int] | None
        :param kwargs:
        :type kwargs: typ.Dict[str, typ.Any]
        :return: train data loader
        :rtype: DataLoader
        """
        if sampler is None:
            param = {
                "shuffle": True,
            }
        else:
            param = {
                "sampler": sampler,
            }

        kwargs = kwargs | param | {"batch_size": batch_size}

        return self.__get_data_loader(
            self.__train_data,
            **kwargs,
        )

    def push_dataloader(self, batch_size, sampler=None, **kwargs):
        """
        Get a data loader for the training push set
        :param batch_size:
        :type batch_size: int
        :param sampler:
        :type sampler: torch.utils.data.Sampler | typ.Iterable[int] | None
        :param kwargs:
        :type kwargs: typ.Dict[str, typ.Any]
        :return: push data loader
        :rtype: DataLoader
        """
        if sampler is None:
            param = {
                "shuffle": False,
            }
        else:
            param = {
                "sampler": sampler,
            }

        kwargs = (
            kwargs
            | param
            | {
                "batch_size": batch_size,
            }
        )

        return self.__get_data_loader(
            self.__push_data,
            **kwargs,
        )

    def test_dataloader(self, batch_size, **kwargs):
        """
        Get a data loader for the testing set
        :param batch_size:
        :type batch_size: int
        :param kwargs:
        :type kwargs: typ.Dict[str, typ.Any]
        :return: test data loader
        :rtype: DataLoader
        """
        kwargs = kwargs | {"batch_size": batch_size, "shuffle": False}

        return self.__get_data_loader(
            self.__test_data,
            **kwargs,
        )


if __name__ == "__main__":
    import os
    from pathlib import Path

    import hydra
    from dotenv import load_dotenv

    from ProtoPNet.util.config_types import Config

    load_dotenv()

    conf_dir = (
        Path(os.getenv("PROJECT_ROOT"))
        / os.getenv("MODULE_NAME")
        / os.getenv("CONFIG_DIR_NAME")
    )

    @hydra.main(version_base=None, config_path=str(conf_dir), config_name="main_config")
    def test(cfg: Config):
        module = CustomDataModule(
            cfg.data.set, "normal_vs_abnormal", cross_validation_folds=5, balanced=True
        )
        for f, (tr, vl) in module.folds:
            ic(f)
            ic(tr)
            ic(vl)

    test()
