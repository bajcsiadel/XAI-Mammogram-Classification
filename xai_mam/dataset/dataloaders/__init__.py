import copy

import albumentations as A
import cv2
import numpy as np
import omegaconf
import pandas as pd
import pipe
import torch
import typing
from albumentations.pytorch import ToTensorV2
from icecream import ic
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets

import xai_mam.utils.config.types as conf_typ
from xai_mam.dataset.metadata import DataFilter
from xai_mam.utils.config._general_types import DatasetConfig
from xai_mam.utils.config._general_types._multifunctional import BatchSize
from xai_mam.utils.config.script_main import Config
from xai_mam.utils.helpers import Augmentations, RepeatedAugmentation
from xai_mam.utils.split_data import stratified_grouped_train_test_split


def _target_transform(target: int) -> torch.Tensor:
    return torch.tensor(target, dtype=torch.long)


def _identity_transform(image: np.ndarray) -> np.ndarray:
    return image


def my_collate_function(batch: tuple):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]


class CustomVisionDataset(datasets.VisionDataset):
    """
    Custom Vision Dataset class for PyTorch.

    :param dataset_meta: Dataset metadata
    :param classification: Classification type
    :param subset: Subset to use. Defaults to ``"train"``.
    :param data_filters: Filters to apply to the data. Defaults to ``None``.
    :param normalize: marks whether to normalize the images or not.
        Defaults to ``True``.
    :param transform: Transform to apply to the images. Defaults to ``None``.
    :param target_transform: Transform to apply to the targets.
        Should return a tensor representing the target.
        Defaults to ``_target_transform``.
    :param debug: flag to mark debug mode, defaults to ``False``
    """
    def __init__(
        self,
        dataset_meta: DatasetConfig,
        classification: str,
        subset: str = "train",
        data_filters: list[DataFilter] | None = None,
        normalize: bool = True,
        transform: Augmentations = None,
        target_transform: typing.Callable[[int], torch.Tensor] = _target_transform,
        debug: bool = False,
    ):
        if transform is None:
            transform = Augmentations()

        if normalize:
            # normalize transform will be applied after ToFloat,
            # which already converts the images between 0 and 1
            normalize_transform = A.Normalize(
                mean=dataset_meta.image_properties.mean,
                std=dataset_meta.image_properties.std,
                max_pixel_value=1.0,
            )
        else:
            normalize_transform = A.NoOp()

        super().__init__(
            str(dataset_meta.image_dir),
            transform=_identity_transform,
            target_transform=target_transform,
        )

        self.__debug = debug
        self.__normalize_transform = normalize_transform

        assert subset in [
            "train",
            "test",
            "all",
        ], "subset must be one of 'train', 'test' or 'all'"
        self.__subset = subset

        self.__dataset_meta = dataset_meta

        metafile_params = dataset_meta.metadata.parameters
        if isinstance(metafile_params, omegaconf.DictConfig):
            metafile_params = omegaconf.OmegaConf.to_object(
                dataset_meta.metadata.parameters
            )
        if (
            isinstance(metafile_params, conf_typ.CSVParametersConfig)
            or type(metafile_params).__name__ == "CSVParameters"
        ):
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
        assert len(cls_types) > 0, (
            f"No classification types found in "
            f"the metadata {dataset_meta.metadata.file}"
        )
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
        self.__classes = np.unique(
            self.__meta_information[(self.__classification, "label")].values
        )
        self.__dataset_meta.number_of_classes = len(self.__classes)
        self.__class_to_number = {cls: i for i, cls in enumerate(self.__classes)}

        self.__raw_targets = self.__meta_information[
            (self.__classification, "label")
        ].to_numpy().repeat(transform.multiplier)
        self.__raw_groups = self.__meta_information.index.get_level_values(
            "patient_id"
        ).to_numpy().repeat(transform.multiplier)

        self.__transform = transform
        self.__target_transform = target_transform

        self.__dataset_meta.input_size = (
            dataset_meta.image_properties.height,
            dataset_meta.image_properties.width,
        )

        self.__remaining_transforms = np.zeros(
            (len(self.__meta_information), len(self.__transform.transforms))
        )
        self.reset_used_transforms()

    def reset_used_transforms(self):
        """
        Reset the transforms as if none have been performed.
        """
        self.__remaining_transforms = [
            copy.deepcopy(self.__transform.get_repetitions())
            for _ in range(len(self.__meta_information))
        ]

    def __compose_transform(
        self, transform: A.Compose | RepeatedAugmentation
    ) -> A.Compose:
        """
        Creating transforms. Surround the transforms with the necessary
        transforms (ToFloat, Normalize, Resize, ToTensor).

        :param transform:
        :return: the final transform
        """
        return A.Compose(
            [
                transform,  # unpack list of transforms
                A.ToFloat(max_value=self.__dataset_meta.image_properties.max_value),
                self.__normalize_transform,
                A.Resize(
                    width=self.__dataset_meta.image_properties.width,
                    height=self.__dataset_meta.image_properties.height,
                ),
                ToTensorV2(),
            ]
        )

    @property
    def class_to_number(self):
        return copy.deepcopy(self.__class_to_number)

    @property
    def targets(self) -> np.ndarray:
        """
        Get the targets of the dataset (after applying the transformations).

        :return: targets of the dataset after transformations
        """
        return self.__raw_targets.copy()

    @property
    def original_targets(self) -> np.ndarray:
        """
        Get the targets of the original dataset (before applying the transformations).

        :return: targets of the original dataset
        """
        return self.__meta_information[(self.__classification, "label")].to_numpy()

    @property
    def groups(self) -> np.ndarray:
        """
        Get the group of each image after the transformations are applied to the
        dataset.

        :return: group of all data after transformations
        """
        return self.__raw_groups.copy()

    @property
    def original_groups(self) -> np.ndarray:
        """
        Get the group of each image before the transformations are applied to the
        dataset.

        :return: group of all data before transformations
        """
        return self.__meta_information.index.get_level_values("patient_id").to_numpy()

    @property
    def multiplier(self) -> int:
        """
        Get the number of repetition of an image as a result of transformations.

        :return: number of repetition of an image
        """
        return self.__transform.multiplier

    @property
    def metadata(self) -> pd.DataFrame:
        """
        Get metadata of the dataset.

        :return: metadata of the dataset
        """
        return self.__meta_information.copy()

    @property
    def dataset_meta(self) -> DatasetConfig:
        """
        Get the dataset information specified in the configuration.

        :return: dataset configuration
        """
        return copy.deepcopy(self.__dataset_meta)

    def debug(self, state: str = "on") -> bool:
        """
        Switch debug mode on and off.

        :param state: if ``"on"``, debug mode is turned on.
            If ``"off"``, debug mode is turned off.
            Defaults to ``"on"``
        :return: current state of debug mode
        """
        self.__debug = state == "on"
        return self.__debug

    def get_original(self, index: int) -> tuple[np.ndarray, int]:
        """
        Get the original image at a specific index from the dataset

        :param index: Index of the sample
        :return: the image and its label at a given index
        """
        sample = self.__meta_information.iloc[index]

        # Define name suffix in case of lesion classification
        suffix = (
            f"-{sample[('mammogram_properties', 'image_number')]}"
            if self.__classification == "benign_vs_malignant"
            else ""
        )
        # Load the image
        image_path = (
            self.__dataset_meta.image_dir
            / f"{sample.name[1]}{suffix}"
              f"{self.__dataset_meta.image_properties.extension}"
        )

        image = (
            np.load(image_path, allow_pickle=True)["image"]
            if image_path.suffix in [".npy", ".npz"]
            else cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        )

        if len(image.shape) > 2:
            # the channel should be the last dimension
            image = image.transpose([1, 2, 0])

        target = self.__class_to_number[sample[self.__classification, "label"]]

        return image, target

    def __repr__(self) -> str:
        """
        Get the representation of the dataset.

        :return: string representation of the dataset
        """
        formatted_transform = "\n\t\t".join(str(self.__transform).split("\n"))

        dataset_ = (
            self.__dataset_meta.name
            if self.__dataset_meta.name != ""
            else "CustomVisionDataset"
        )
        return (
            f"CustomVisionDataset(\n"
            f"\tname={dataset_},\n"
            f"\tsubset={self.__subset},\n"
            f"\tn_samples={len(self)},\n"
            f"\tlocation={self.__dataset_meta.image_dir},\n"
            f"\ttransform={formatted_transform},\n"
            f")"
        )

    def __len__(self) -> int:
        """
        Get the number of images in the dataset.

        :return: number of images
        """
        return len(self.__meta_information) * self.__transform.multiplier

    def __getitem__(self, index: int) -> [torch.Tensor, torch.Tensor]:
        """
        Get sample at a specific index from the dataset.

        :param index: index of the sample
        :return: the image and its label at a given index
        """
        sample_index = index // self.__transform.multiplier

        # Load the image
        image, target = self.get_original(sample_index)

        if self.__transform.multiplier > 1:
            # apply transforms
            remaining_transform_indices = np.where(
                self.__remaining_transforms[sample_index] > 0
            )[0]
            # convert the index to int, otherwise it will be np.int64
            selected_transform_index = int(
                np.random.choice(remaining_transform_indices)
            )
            self.__remaining_transforms[sample_index][selected_transform_index] -= 1

            current_transform = self.__transform.transforms[selected_transform_index]
        else:
            current_transform = self.__transform.transforms[0]
        final_transform = self.__compose_transform(current_transform)

        image = final_transform(image=image)["image"]
        target = self.__target_transform(target)

        return image, target


class CustomDataModule:
    """
    DataModule to define data loaders.

    :param data:
    :param classification:
    :param data_filters: Filters to apply to the data. Defaults to ``None``.
    :param cross_validation_folds: Number of cross validation folds.
        Defaults to ``None``.
    :param stratified: Defaults to ``False``.
    :param balanced: Defaults to ``False``.
    :param grouped: Defaults to ``False``.
    :param n_workers: Defaults to ``0``.
    :param seed: Defaults to ``0``.
    :param debug: Defaults to ``False``.
    :param batch_size: Defaults to ``None``.
    """
    def __init__(
        self,
        data: DatasetConfig,
        classification: str,
        data_filters: list[typing.Callable[[pd.DataFrame], pd.DataFrame]] | None = None,
        cross_validation_folds: int | None = None,
        stratified: bool = False,
        balanced: bool = False,
        grouped: bool = False,
        n_workers: int = 0,
        seed: int = 0,
        debug: bool = False,
        batch_size: BatchSize = None,
    ):
        if batch_size is None:
            batch_size = conf_typ.BatchSize(32, 16)

        self.__data = data
        dataset_params = {
            "dataset_meta": self.__data,
            "classification": classification,
            "data_filters": data_filters,
        }

        # define datasets
        self.__train_data = CustomVisionDataset(
            **dataset_params,
            transform=Augmentations(
                transforms=self.__data.image_properties.augmentations.train
            ),
            subset="train",
        )
        self.__validation_data = CustomVisionDataset(
            **dataset_params,
            transform=Augmentations(
                transforms=self.__data.image_properties.augmentations.train
            ),
            subset="train",
        )
        self.__push_data = CustomVisionDataset(
            **dataset_params,
            transform=Augmentations(self.__data.image_properties.augmentations.push),
            normalize=False,
            subset="train",
        )
        self.__test_data = CustomVisionDataset(
            **dataset_params,
            subset="test",
        )

        self.__n_workers = n_workers
        self.__debug = debug

        self.__init_cross_validation(
            cross_validation_folds, stratified, balanced, grouped, batch_size, seed
        )

    def __init_cross_validation(
        self,
        cross_validation_folds: int,
        stratified: bool,
        balanced: bool,
        groups: bool,
        batch_size: BatchSize,
        seed: int,
    ):
        """
        Initialize cross validation folds.

        :param cross_validation_folds: number of cross validation folds
        :param stratified:
        :param balanced:
        :param groups:
        :param batch_size:
        :param seed:
        """
        if self.__debug or cross_validation_folds in [None, 0, 1]:
            debug_specific_params = {}
            if self.__debug:
                debug_specific_params = {
                    "test_size": batch_size.validation,
                    "train_size": batch_size.train,
                }
            # else:
            #     debug_specific_params = {
            #         "test_size": 0.4,
            #     }
            train_idx, validation_idx = stratified_grouped_train_test_split(
                self.__train_data.metadata,
                self.__train_data.original_targets,
                self.__train_data.original_groups,
                **debug_specific_params,
                random_state=seed,
            )

            # if not self.__debug:
            #     train_idx, validation_idx = validation_idx, train_idx
            # train_idx = np.array([0, 1, 4, 5, 6, 8, 9, 10, 12, 13, 14, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 52, 54, 55, 56, 57, 58, 59, 61, 62, 63, 65, 66, 67, 69, 70, 71, 73, 74, 75, 76, 77, 78, 79])
            # validation_idx = np.array([2, 3, 7, 11, 17, 18, 28, 29, 39, 43, 51, 53, 60, 64, 68, 72, 80])

            self.__fold_generator = [(train_idx, validation_idx)]
            return

        targets = self.__train_data.targets
        sample_groups = self.__train_data.groups

        cv_kwargs = {}
        if groups or balanced:
            cv_kwargs["groups"] = sample_groups
        if stratified or balanced:
            cv_kwargs["y"] = targets

        if balanced:
            from xai_mam.utils.split_data.cross_validation import BalancedGroupKFold

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

        self.__fold_generator = cross_validator.split(
            self.__train_data.metadata, **cv_kwargs
        )

    @property
    def debug(self) -> bool:
        """
        Get debug mode of the module.

        :return: debug mode
        """
        return self.__debug

    @property
    def dataset(self) -> DatasetConfig:
        return copy.deepcopy(self.__data)

    @property
    def train_data(self) -> CustomVisionDataset:
        return self.__train_data

    @property
    def validation_data(self) -> CustomVisionDataset:
        return self.__validation_data or self.__train_data

    @property
    def push_data(self) -> CustomVisionDataset:
        return self.__push_data

    @property
    def test_data(self) -> CustomVisionDataset:
        return self.__test_data

    @property
    def folds(
        self
    ) -> typing.Iterator[tuple[int, (SubsetRandomSampler, SubsetRandomSampler)]]:
        """
        Generate the folds and the corresponding samplers.

        :return: fold number, (train sampler, validation sampler)
        """
        for fold, (train_idx, validation_idx) in enumerate(
            self.__fold_generator, start=1
        ):
            if self.__train_data.multiplier > 1:
                train_idx = np.array(
                    [
                        range(
                            index * self.__train_data.multiplier,
                            (index + 1) * self.__train_data.multiplier,
                        )
                        for index in train_idx
                    ]
                ).flatten()
            if self.__validation_data.multiplier > 1:
                validation_idx = np.array(
                    [
                        range(
                            index * self.__validation_data.multiplier,
                            (index + 1) * self.__validation_data.multiplier,
                        )
                        for index in validation_idx
                    ]
                ).flatten()
            yield fold, (
                SubsetRandomSampler(train_idx),
                SubsetRandomSampler(validation_idx),
            )

    def __get_data_loader(self, dataset: CustomVisionDataset, **kwargs) -> DataLoader:
        """
        Get a data loader for a given dataset
        :param dataset:
        :param kwargs:
        :return: data loader
        """
        return DataLoader(dataset, num_workers=self.__n_workers, **kwargs)

    def train_dataloader(
        self,
        batch_size: int,
        sampler: torch.utils.data.Sampler | typing.Iterable[int] = None,
        **kwargs
    ) -> DataLoader:
        """
        Get a data loader for the training set.

        :param batch_size:
        :param sampler:
        :param kwargs:
        :return: train data loader
        """
        if sampler is None:
            param = {
                "shuffle": True,
            }
        else:
            param = {
                "sampler": sampler,
            }

        if self.__debug:
            batch_size = len(self.__train_data)

        kwargs = kwargs | param | {"batch_size": batch_size}

        return self.__get_data_loader(
            self.__train_data,
            **kwargs,
        )

    def validation_dataloader(
        self,
        batch_size: int,
        sampler: torch.utils.data.Sampler | typing.Iterable[int] = None,
        **kwargs
    ) -> DataLoader:
        """
        Get a data loader for the validation set.

        :param batch_size:
        :param sampler:
        :param kwargs:
        :return: validation data loader
        """
        if sampler is None:
            param = {
                "shuffle": False,
            }
        else:
            param = {
                "sampler": sampler,
            }

        if self.__debug:
            batch_size = len(self.__validation_data)

        kwargs = kwargs | param | {"batch_size": batch_size}

        return self.__get_data_loader(
            self.__validation_data or self.__train_data,
            **kwargs,
        )

    def push_dataloader(
        self,
        batch_size: int,
        sampler: SubsetRandomSampler | typing.Iterable[int] = None,
        **kwargs,
    ) -> DataLoader:
        """
        Get a data loader for the training push set.

        :param batch_size:
        :param sampler:
        :param kwargs:
        :return: push data loader
        """
        if sampler is None:
            param = {
                "shuffle": False,
            }
        else:
            if self.__train_data.multiplier > 1:
                if isinstance(sampler, torch.utils.data.SubsetRandomSampler):
                    indices = sampler.indices
                else:
                    indices = copy.deepcopy(sampler)
                sampler = torch.utils.data.sampler.SubsetRandomSampler(
                    np.array(indices) // self.__train_data.multiplier
                )
            param = {
                "sampler": sampler,
            }

        if self.__debug:
            batch_size = len(self.__train_data)

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

    def test_dataloader(self, batch_size: int, **kwargs) -> DataLoader:
        """
        Get a data loader for the testing set
        :param batch_size:
        :param kwargs:
        :return: test data loader
        """
        kwargs = kwargs | {"batch_size": batch_size, "shuffle": False}

        return self.__get_data_loader(
            self.__test_data,
            **kwargs,
        )


if __name__ == "__main__":
    import hydra
    import matplotlib.pyplot as plt

    from xai_mam.utils.environment import get_env

    Config.init_store()

    @hydra.main(
        version_base=None,
        config_path=get_env("CONFIG_PATH"),
        config_name="main_config",
    )
    def test(cfg: Config):
        cfg = omegaconf.OmegaConf.to_object(cfg)
        module = hydra.utils.instantiate(cfg.data.datamodule)
        for f, (tr, vl) in enumerate(module.folds, start=1):
            ic(f)

            ic(tr)
            if tr is not None:
                ic(len(tr))

            ic(vl)
            if vl is not None:
                ic(len(vl))

        data = module.train_data
        data.debug()

        _, axs = plt.subplots(1, 2)

        image, label = data[0]
        axs[0].imshow(image.squeeze(), cmap="gray")
        axs[0].set_title("Augmented Image")
        axs[0].axis("off")
        ic("Original")
        ic(image.min())
        ic(image.max())
        ic(image.std())
        ic(image.mean())

        image, label = data.get_original(0)
        axs[1].imshow(image.squeeze(), cmap="gray")
        axs[1].set_title("Original Image")
        axs[1].axis("off")
        ic("Augmented")
        ic(image.min())
        ic(image.max())
        ic(image.std())
        ic(image.mean())
        plt.show()

        # in _data_config set:
        # image_properties:
        #     extension: .npz
        #     augmentations:
        #         train:
        #             transforms:
        #                 - _target_: xai_mam.utils.helpers.RepeatedAugmentation
        #                 transforms:
        #                     - _target_: albumentations.Rotate
        #                     limit: 45
        #                     border_mode: 0
        #                     crop_border: true
        #                     p: 1.0
        #                 n: 8
        _, axs = plt.subplots(3, 3)
        for i in range(9):
            image, _ = data[i]
            axs[i // 3, i % 3].imshow(image.squeeze(), cmap="gray")
            axs[i // 3, i % 3].set_title(f"{i + 1}")
            axs[i // 3, i % 3].axis("off")
        plt.show()

    test()
