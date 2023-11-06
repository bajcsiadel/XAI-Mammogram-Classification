if __name__ == '__main__':
    from dotenv import load_dotenv

    load_dotenv()

    from ProtoPNet.dataset.metadata import DATASETS

import albumentations as A
from albumentations.pytorch import ToTensorV2

import cv2
import numpy as np
import os
import pandas as pd

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets

from ProtoPNet.dataset.metadata import DatasetInformation


def _target_transform(target):
    return torch.tensor(target, dtype=torch.long)


class CustomVisionDataset(datasets.VisionDataset):
    """
    Custom Vision Dataset class for PyTorch.

    :param dataset_meta: Dataset metadata
    :type dataset_meta: DatasetInformation
    :param classification: Classification type
    :type classification: str
    :param subset: Subset to use
    :type subset: str
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
            normalize=True,
            transform=A.NoOp(),
            target_transform=_target_transform
    ):
        has_transforms = True
        if isinstance(transform, A.NoOp):
            has_transforms = False

        if isinstance(transform, A.BasicTransform):
            transform = [transform]
        if normalize:
            transform.append(A.Normalize(mean=dataset_meta.USED_IMAGES.MEAN, std=dataset_meta.USED_IMAGES.STD, max_pixel_value=1.0))

        transform = A.Compose([
            A.ToFloat(max_value=dataset_meta.IMAGE_PROPERTIES.MAX_VALUE),
            *transform,  # unpack list of transforms
            ToTensorV2(),
        ])

        super().__init__(dataset_meta.USED_IMAGES.DIR, transform=transform, target_transform=target_transform)

        # assert subset in ["train", "test"], "subset must be one of 'train' or 'test'"
        self.__subset = subset

        self.__dataset_meta = dataset_meta

        self.__meta_information = pd.read_csv(
            dataset_meta.METADATA.FILE,
            **dataset_meta.METADATA.PARAMETERS
        )

        assert isinstance(self.__meta_information.columns, pd.MultiIndex), "metadata does not have split information"
        cls_types = list(filter(lambda column: "_vs_" in column, self.__meta_information.columns.get_level_values(0)))
        assert len(cls_types) > 0, f"no classification types found in the metadata {dataset_meta.METADATA.FILE}"
        assert classification in f"cls_types, classification must be one from {cls_types}"
        self.__classification = classification

        # Filter out the rows that are not in the subset
        if subset != "all":
            self.__meta_information = self.__meta_information[
                self.__meta_information[(self.__classification, "subset")] == self.__subset
            ]
        self.__classes = self.__meta_information[(self.__classification, "label")].unique().tolist()
        self.__number_of_classes = len(self.__classes)
        self.__class_to_number = {cls: i for i, cls in enumerate(self.__classes)}

        self.__has_transforms = has_transforms
        self.__transform = transform

        # define imbalance of dataset
        self.__imbalance = self.__meta_information.groupby((self.__classification, "label")).size().to_dict()

        self.__target_transform = target_transform

        self.__current_image = None
        self.__current_target = None

    @property
    def targets(self):
        return self.__meta_information[(self.__classification, "label")].copy()

    @property
    def metadata(self):
        return self.__meta_information.copy()

    def __repr__(self):
        formatted_transform = "\n\t".join(str(self.__transform).split("\n"))
        return (
            f"Dataset {self.__dataset_meta.NAME if self.__dataset_meta.NAME != '' else 'CustomVisionDataset'}\n"
            f"\tSubset: {self.__subset}\n"
            f"\tNumber of datapoints: {len(self)}\n"
            f"\tImage location: {self.__dataset_meta.USED_IMAGES.DIR}\n"
            f"\tTransform: {formatted_transform}\n"
        )

    def __len__(self):
        """
        Get the number of images in the dataset

        :return: number of images
        :rtype: int
        """
        # TODO: if the dataset is imbalanced, solve it by augmentation and recompute the number of samples
        return len(self.__meta_information)

    def __getitem__(self, index):
        """
        Get sample at a specific index from the dataset

        :param index: Index of the sample
        :type index: int
        :return: Return the image and its label at a given index
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
        # TODO: if the dataset is imbalanced, solve it by augmentation
        # TODO: compute index correspondingly
        sample = self.__meta_information.iloc[index]

        # Load the image
        image_path = os.path.join(
            self.__dataset_meta.USED_IMAGES.DIR,
            f"{sample.name[1]}{self.__dataset_meta.IMAGE_PROPERTIES.EXTENSION}"
        )
        if self.__dataset_meta.IMAGE_PROPERTIES.EXTENSION in [".npy", ".npz"]:
            # quicker to load than cv2.imread
            self.__current_image = np.load(image_path)["image"]
        else:
            self.__current_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        target = self.__class_to_number[sample[self.__classification, "label"]]
        self.__current_target = self.__target_transform(target)

        if self.__has_transforms:
            # TODO: apply augmentation
            image = None
        else:
            image = self.__transform(image=self.__current_image)["image"]

        return image, self.__current_target


class CustomTrainDataLoader:
    def __init__(
            self,
            dataset,
            cross_validation_folds,
            stratified=True,
            groups=True,
            seed=None,
            **kwargs
    ):
        self.__dataset = dataset
        self.__targets = dataset.targets
        self.__groups = dataset.metadata.index.get_level_values("patient_id").to_numpy() if groups else None

        self.__data_loader_kwargs = kwargs

        self.__cross_validation_folds = cross_validation_folds

        if cross_validation_folds <= 1:
            raise ValueError(f"cross_validation_folds must be greater than 1, not {cross_validation_folds}")

        if stratified and groups:
            from sklearn.model_selection import StratifiedGroupKFold
            cross_validator_class = StratifiedGroupKFold
            self.__cv_kwargs = {
                "y": self.__targets,
                "groups": self.__groups
            }
        elif stratified and not groups:
            from sklearn.model_selection import StratifiedKFold
            cross_validator_class = StratifiedKFold
            self.__cv_kwargs = {
                "y": self.__targets
            }
        elif not stratified and groups:
            from sklearn.model_selection import GroupKFold
            cross_validator_class = GroupKFold
            self.__cv_kwargs = {
                "groups": self.__groups
            }
        else:
            # not stratified and not groups
            from sklearn.model_selection import KFold
            cross_validator_class = KFold
            self.__cv_kwargs = {}

        self.__cross_validator = cross_validator_class(
            n_splits=cross_validation_folds,
            shuffle=True,
            random_state=seed
        )

    @property
    def cross_validation_folds(self):
        return self.__cross_validation_folds

    def __len__(self):
        return len(self.__dataset)

    def __iter__(self):
        for fold, (train_idx, val_idx) in enumerate(self.__cross_validator.split(self.__dataset, **self.__cv_kwargs)):
            train_sampler = SubsetRandomSampler(train_idx)
            validation_sampler = SubsetRandomSampler(val_idx)
            yield fold, (
                DataLoader(self.__dataset, sampler=train_sampler, **self.__data_loader_kwargs),
                DataLoader(self.__dataset, sampler=validation_sampler, **self.__data_loader_kwargs)
            )


class CustomDataModule:
    """
    DataModule to define data loaders.

    :param data:
    :type data: DatasetInformation
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
    :param dataset_class:
    """
    def __init__(
            self,
            data,
            used_images,
            classification,
            batch_size,
            cross_validation_folds,
            stratified=True,
            groups=True,
            push_batch_size=None,
            num_workers=0,
            seed=None,
            dataset_class=None,
    ):
        assert issubclass(dataset_class, CustomVisionDataset), f"dataset_class must be a subclass of CustomVisionDataset, not {type(dataset_class)}"

        self.__data = data
        if self.__data.USED_IMAGES is None:
            if used_images in data.VERSIONS:
                self.__data.USED_IMAGES = data.VERSIONS[used_images]
            else:
                raise ValueError(f"'used_images' must be one of {list(data.VERSIONS.keys())}, not {used_images}")

        if push_batch_size is None:
            push_batch_size = batch_size

        if dataset_class is None:
            dataset_class = CustomVisionDataset
            dataset_params = {
                "dataset_meta": self.__data,
                "classification": classification,
            }
        else:
            dataset_params = {
                "classification": classification,
            }
        train_data = dataset_class(**dataset_params, subset="train")
        push_data = dataset_class(**dataset_params, normalize=False, subset="train")
        test_data = dataset_class(**dataset_params, subset="test")

        self.__train_loader = CustomTrainDataLoader(
            train_data,
            cross_validation_folds,
            stratified,
            groups,
            seed=seed,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        self.__push_loader = DataLoader(
            push_data,
            batch_size=push_batch_size,
            num_workers=num_workers,
            shuffle=False
        )
        self.__test_loader = DataLoader(
            test_data,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False
        )

    def train_dataloader(self):
        return self.__train_loader

    def push_dataloader(self):
        return self.__push_loader

    def test_dataloader(self):
        return self.__test_loader


if __name__ == '__main__':
    from dotenv import load_dotenv

    load_dotenv()
    ds = DATASETS["MIAS"]
    ds.USED_IMAGES = ds.VERSIONS["original"]

    module = CustomDataModule(ds, "original", "normal_vs_abnormal", 32, 5)
    for fold, (tr, vl) in module.train_dataloader():
        print(fold)
        print(tr)
        print(vl)

    # loader = CustomDataModule(ds, "original", "normal_vs_abnormal", 32)
    # tr_loader = loader.train_dataloader()
    # from sklearn.model_selection import StratifiedGroupKFold
    #
    # skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    # for fold, (tr_idx, val_idx) in enumerate(skf.split(tr_loader)):
    #     print(fold)
    #     print(tr_idx)
    #     print(val_idx)

