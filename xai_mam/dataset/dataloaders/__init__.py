import copy

import albumentations as A
import cv2
import numpy as np
import omegaconf
import pandas as pd
import pipe
import torch
from albumentations.pytorch import ToTensorV2
from icecream import ic
from torch.utils.data import DataLoader
from torchvision import datasets

import xai_mam.utils.config.types as conf_typ
from xai_mam.utils.config.script_main import Config, init_config_store
from xai_mam.utils.helpers import Augmentations
from xai_mam.utils.split_data import stratified_grouped_train_test_split


def _target_transform(target):
    return torch.tensor(target, dtype=torch.long)


def _identity_transform(image):
    return image


def my_collate_function(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]


class CustomVisionDataset(datasets.VisionDataset):
    """
    Custom Vision Dataset class for PyTorch.

    :param dataset_meta: Dataset metadata
    :type dataset_meta: xai_mam.util.config_types.Dataset
    :param classification: Classification type
    :type classification: str
    :param subset: Subset to use
    :type subset: str
    :param data_filters: Filters to apply to the data
    :type data_filters: list[ProtoPNet.dataset.metadata.DataFilter] | None
    :param transform: Transform to apply to the images
    :type transform: xai_mam.utils.helpers.Augmentations
    :param target_transform: Transform to apply to the targets.
        Should return a tensor representing the target.
    :type target_transform: typ.Callable[[], torch.Tensor]
    :param debug: flag to mark debug mode, defaults to False
    :type debug: bool
    """

    def __init__(
        self,
        dataset_meta,
        classification,
        subset="train",
        data_filters=None,
        normalize=True,
        transform=None,
        target_transform=_target_transform,
        debug=False,
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
        self.__classes = (
            self.__meta_information[(self.__classification, "label")].unique().tolist()
        )
        self.__dataset_meta.number_of_classes = len(self.__classes)
        self.__class_to_number = {cls: i for i, cls in enumerate(self.__classes)}

        self.__transform = transform
        self.__target_transform = target_transform

        self.__dataset_meta.input_size = (
            dataset_meta.image_properties.height,
            dataset_meta.image_properties.width,
        )

        self.__remaining_transforms = [
            copy.deepcopy(self.__transform.get_repetitions())
            for _ in range(len(self.__meta_information))
        ]

    def __compose_transform(self, transform):
        """
        Creating transforms. Surround the transforms with the necessary
        transforms (ToFloat, Normalize, Resize, ToTensor).

        :param transform:
        :type transform: albumentations.Compose |
        xai_mam.utils.helpers.RepeatedAugmentation
        :return: the final transform
        :rtype: albumentations.Compose
        """
        return A.Compose(
            [
                A.ToFloat(max_value=self.__dataset_meta.image_properties.max_value),
                transform,  # unpack list of transforms
                self.__normalize_transform,
                A.Resize(
                    width=self.__dataset_meta.image_properties.width,
                    height=self.__dataset_meta.image_properties.height,
                ),
                ToTensorV2(),
            ]
        )

    @property
    def targets(self):
        """
        Get the targets of the dataset

        :return:
        :rtype: np.ndarray
        """
        return self.__meta_information[(self.__classification, "label")].to_numpy()

    @property
    def groups(self):
        return self.__meta_information.index.get_level_values("patient_id").to_numpy()

    @property
    def multiplier(self):
        return self.__transform.multiplier

    @property
    def metadata(self):
        """
        Get metadata of the dataset

        :return:
        :rtype: pd.DataFrame
        """
        return self.__meta_information.copy()

    def debug(self, state="on"):
        """
        Switch debug mode on and off.

        :param state: if ``"on"``, debug mode is turned on.
            If ``"off"``, debug mode is turned off.
            Defaults to ``"on"``
        :type state: str
        :return: current state of debug mode
        :rtype: bool
        """
        self.__debug = state == "on"
        return self.__debug

    def get_original(self, index):
        """
        Get the original image at a specific index from the dataset

        :param index: Index of the sample
        :type index: int
        :return: Return the image and its label at a given index
        :rtype: (numpy.ndarray, int)
        """
        sample = self.__meta_information.iloc[index]

        # Define name suffix in case of lesion classification
        suffix = f"-{sample[('mammogram_properties', 'image_number')]}" \
            if self.__classification == "benign_vs_malignant" else ""
        # Load the image
        image_path = (
            self.__dataset_meta.image_dir
            / f"{sample.name[1]}{suffix}{self.__dataset_meta.image_properties.extension}"
        )

        image = (
            np.load(image_path, allow_pickle=True)["image"]
            if image_path.suffix in [".npy", ".npz"]
            else cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        )

        target = self.__class_to_number[sample[self.__classification, "label"]]

        return image, target

    def __repr__(self):
        """
        Get the representation of the dataset

        :return:
        :rtype: str
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

    def __len__(self):
        """
        Get the number of images in the dataset

        :return: number of images
        :rtype: int
        """
        return len(self.__meta_information) * self.__transform.multiplier

    def __getitem__(self, index):
        """
        Get sample at a specific index from the dataset

        :param index: Index of the sample
        :type index: int
        :return: Return the image and its label at a given index
        :rtype: typ.Tuple[torch.Tensor, torch.Tensor]
        """
        sample_index = index // self.__transform.multiplier

        # Load the image
        image, target = self.get_original(sample_index)

        # apply transforms
        remaining_transform_indices = np.where(
            self.__remaining_transforms[sample_index] > 0
        )[0]
        # convert the index to int, otherwise it will be np.int64
        selected_transform_index = int(np.random.choice(remaining_transform_indices))
        self.__remaining_transforms[sample_index][selected_transform_index] -= 1

        current_transform = self.__transform.transforms[selected_transform_index]
        final_transform = self.__compose_transform(current_transform)
        image = final_transform(image=image)["image"]
        target = self.__target_transform(target)

        return image, target


class CustomSubset(torch.utils.data.Subset):
    """
    Custom Subset class for PyTorch containing `targets` and `groups` properties.

    :param dataset: root dataset
    :type dataset: CustomVisionDataset
    :param indices: indices of the included samples
    :type indices: typ.Sequence[int]
    """

    def __init__(self, dataset, indices):
        self.__patient_indices = copy.deepcopy(indices)
        if dataset.multiplier > 1:
            indices = np.array(
                [
                    range(
                        index * dataset.multiplier,
                        (index + 1) * dataset.multiplier,
                    )
                    for index in indices
                ]
            ).flatten()
        super().__init__(dataset, indices)

    @property
    def metadata(self):
        if hasattr(self.dataset, "metadata"):
            return self.dataset.metadata.iloc[self.__patient_indices]
        return pd.DataFrame()

    @property
    def targets(self):
        if hasattr(self.dataset, "targets"):
            return self.dataset.targets[self.__patient_indices]
        return []

    @property
    def groups(self):
        if hasattr(self.dataset, "groups"):
            return self.dataset.groups[self.__patient_indices]
        return []

    @property
    def multiplier(self):
        if hasattr(self.dataset, "multiplier"):
            return self.dataset.multiplier
        return 1

    def __repr__(self):
        return (
            f"CustomSubset(\n"
            f"\tdataset={self.dataset},\n"
            f"\tindices={self.indices},\n"
            f"\tmin(indices)={min(*self.indices)},\n"
            f"\tmax(indices)={max(*self.indices)},\n"
            f")"
        )


class CustomDataModule:
    """
    DataModule to define data loaders.

    :param data:
    :type data: Dataset
    :param classification:
    :type classification: str
    :param data_filters: Filters to apply to the data
    :type data_filters: list[(pd.DataFrame) -> pd.DataFrame] | None
    :param cross_validation_folds: Number of cross validation folds
    :type cross_validation_folds: int
    :param stratified:
    :type stratified: bool
    :param balanced:
    :type balanced: bool
    :param grouped:
    :type grouped: bool
    :param num_workers:
    :type num_workers: int
    :param seed:
    :type seed: int
    :param debug:
    :type debug: bool
    :param batch_size:
    :type batch_size: conf_typ.BatchSize
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
        debug=False,
        batch_size=None,
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

        self.__number_of_workers = num_workers
        self.__debug = debug

        if self.__debug or cross_validation_folds in [None, 0, 1]:
            debug_specific_params = {}
            if self.__debug:
                debug_specific_params = {
                    "test_size": batch_size.validation,
                    "train_size": batch_size.train,
                }
            train_idx, validation_idx = stratified_grouped_train_test_split(
                self.__train_data.metadata,
                self.__train_data.targets,
                self.__train_data.groups,
                **debug_specific_params,
                random_state=seed,
            )

            self.__validation_data = CustomSubset(
                self.__validation_data, validation_idx
            )
            # using the original image indices
            self.__push_data = CustomSubset(self.__push_data, train_idx)
            # using the image indices considering the augmentation
            self.__train_data = CustomSubset(self.__train_data, train_idx)

        self.__init_cross_validation(
            cross_validation_folds, stratified, balanced, grouped, seed
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
        if cross_validation_folds in [None, 0, 1] or self.__debug:
            self.__fold_generator = [(None, None)]
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

        self.__fold_generator = cross_validator.split(self.__train_data, **cv_kwargs)

    @property
    def debug(self):
        return self.__debug

    @property
    def dataset(self):
        return copy.deepcopy(self.__data)

    @property
    def train_data(self):
        return self.__train_data

    @property
    def validation_data(self):
        return self.__validation_data or self.__train_data

    @property
    def push_data(self):
        return self.__push_data

    @property
    def test_data(self):
        return self.__test_data

    @property
    def folds(self):
        """
        Generate the folds and the corresponding samplers

        :return: fold number, (train sampler, validation sampler)
        :rtype: typing.Generator[
            int,
            tuple[
                torch.utils.data.SubsetRandomSampler,
                torch.utils.data.SubsetRandomSampler
            ]]
        """
        yield from self.__fold_generator

    def log_data_information(self, logger):
        """
        Log information about the data

        :param logger:
        :type logger: ProtoPNet.utils.log.Log
        """
        CustomDataModule.__log_data(logger, self.__train_data, "train")
        if self.__validation_data is not None:
            CustomDataModule.__log_data(logger, self.__validation_data, "validation")
        CustomDataModule.__log_data(logger, self.__push_data, "push")

    @staticmethod
    def __log_data(logger, data, name):
        """
        Log information about a dataset
        :param logger:
        :type logger: ProtoPNet.utils.log.Log
        :param data:
        :type data: CustomSubset | CustomVisionDataset
        :param name:
        :type name: str
        """
        logger.info(f"{name}")
        logger.increase_indent()
        logger.info(f"size: {len(data)} ({len(data.targets)} x {data.multiplier})")
        logger.info("distribution:")
        logger.increase_indent()
        distribution = pd.DataFrame(columns=["count", "perc"])
        classes = np.unique(data.targets)
        for cls in classes:
            count = np.sum(data.targets == cls) * data.multiplier
            distribution.loc[cls] = [count, count / len(data)]
        distribution["count"] = distribution["count"].astype("int")
        logger.info(f"{distribution.to_string(formatters={'perc': '{:.2%}'.format})}")
        logger.decrease_indent(times=2)

    def __get_data_loader(self, dataset, **kwargs):
        """
        Get a data loader for a given dataset
        :param dataset:
        :type dataset: CustomVisionDataset | CustomSubset
        :param kwargs:
        :type kwargs: dict[str, typing.Any]
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
        :type sampler: torch.utils.data.Sampler | typing.Iterable[int] | None
        :param kwargs:
        :type kwargs: dict[str, typing.Any]
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

        if self.__debug:
            batch_size = len(self.__train_data)

        kwargs = kwargs | param | {"batch_size": batch_size}

        return self.__get_data_loader(
            self.__train_data,
            **kwargs,
        )

    def validation_dataloader(self, batch_size, sampler=None, **kwargs):
        """
        Get a data loader for the validation set
        :param batch_size:
        :type batch_size: int
        :param sampler:
        :type sampler: torch.utils.data.Sampler | typing.Iterable[int] | None
        :param kwargs:
        :type kwargs: dict[str, typing.Any]
        :return: validation data loader
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

        if self.__debug:
            batch_size = len(self.__validation_data)

        kwargs = kwargs | param | {"batch_size": batch_size}

        return self.__get_data_loader(
            self.__validation_data or self.__train_data,
            **kwargs,
        )

    def push_dataloader(self, batch_size, sampler=None, **kwargs):
        """
        Get a data loader for the training push set
        :param batch_size:
        :type batch_size: int
        :param sampler:
        :type sampler: torch.utils.data.Sampler | typing.Iterable[int] | None
        :param kwargs:
        :type kwargs: dict[str, typing.Any]
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

    def test_dataloader(self, batch_size, **kwargs):
        """
        Get a data loader for the testing set
        :param batch_size:
        :type batch_size: int
        :param kwargs:
        :type kwargs: dict[str, typing.Any]
        :return: test data loader
        :rtype: DataLoader
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

    init_config_store()

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
