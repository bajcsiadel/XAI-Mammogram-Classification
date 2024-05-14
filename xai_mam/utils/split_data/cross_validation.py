import numpy as np
from icecream import ic
from sklearn.model_selection import GroupKFold


class BalancedGroupKFold(GroupKFold):
    """
    K-fold iterator variant with non-overlapping groups and balanced class distribution.
    The smallest class samples are used in each fold. The other classes are used to fill the fold,
    with the same number of samples from each class

    :param n_splits:
    :type n_splits: int
    :param random_state:
    :type random_state: int
    """

    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        super(BalancedGroupKFold, self).__init__(n_splits=n_splits)

        self.__random_state = random_state
        self.__shuffle = shuffle
        np.random.seed(random_state)

        self.__smallest_class = None
        self.__class_information = {}
        self.__count_use = None

        self.__train_size = None
        self.__test_size = None

        self.__cv_smallest_class = GroupKFold(n_splits=n_splits)

    def __init_data(self, y, groups):
        if groups is None:
            raise ValueError(f"groups in split with BalancedGroupKFold can not be None")

        unique_classes, sample_counts = np.unique(y, return_counts=True)
        smallest_class_index = np.argmin(sample_counts)
        self.__smallest_class = unique_classes[smallest_class_index]

        self.__count_use = {}

        self.__test_size = np.ceil(sample_counts[smallest_class_index] / self.n_splits)
        self.__train_size = (self.n_splits - 1) * self.__test_size

        for unique_class, samples in zip(unique_classes, sample_counts):
            indices = np.where(y == unique_class)[0]

            sample_mask = np.zeros_like(y, dtype=np.bool_)
            sample_mask[indices] = True

            unique_groups, samples_in_group = np.unique(
                groups[indices], return_counts=True
            )

            shuffle_indices = np.arange(len(unique_groups))

            if self.__shuffle:
                np.random.shuffle(shuffle_indices)

            multiplier = np.ceil(
                sample_counts[smallest_class_index] * self.n_splits / samples
            ).astype(np.uint8)

            shuffle_indices = np.tile(shuffle_indices, multiplier)

            self.__class_information[unique_class] = {
                "sample_indices": indices,
                "sample_mask": sample_mask,
                "sample_group": groups[indices],
                "groups": unique_groups[shuffle_indices],
                "samples_in_group": samples_in_group[shuffle_indices],
                "start_index": 0,
                "extra_samples": np.argmin(
                    np.abs(
                        np.cumsum(samples_in_group[shuffle_indices])
                        - self.n_splits * sample_counts[smallest_class_index]
                    )
                ),
            }

        for subset in ["train", "test"]:
            self.__count_use[subset] = np.zeros_like(y, dtype=np.uint8)

    def split(self, X, y=None, groups=None):
        if y is None:
            raise ValueError("y cannot be None")
        else:
            y = np.array(y)

        if self.__smallest_class is None:
            self.__init_data(y, groups)

        for train_idx, test_idx in self.__cv_smallest_class.split(
            self.__class_information[self.__smallest_class]["sample_indices"],
            groups=self.__class_information[self.__smallest_class]["sample_group"],
        ):
            # define train and test size
            train_sample_size = len(train_idx)
            test_sample_size = len(test_idx)

            # get groups of train and test samples
            train_groups = np.unique(
                self.__class_information[self.__smallest_class]["sample_group"][
                    train_idx
                ]
            )
            test_groups = np.unique(
                self.__class_information[self.__smallest_class]["sample_group"][
                    test_idx
                ]
            )

            # map train and test indices to original indices
            train_idx = self.__class_information[self.__smallest_class][
                "sample_indices"
            ][train_idx]
            test_idx = self.__class_information[self.__smallest_class][
                "sample_indices"
            ][test_idx]

            for unique_class, class_information in self.__class_information.items():
                if unique_class == self.__smallest_class:
                    # skip the split of the smallest class
                    # it is already done
                    continue

                train_idx, train_groups = self.__resolve_group_overlap(
                    class_information,
                    train_groups,
                    train_sample_size,
                    test_groups,
                    train_idx,
                    self.__count_use["train"],
                )

                test_idx, test_groups = self.__resolve_group_overlap(
                    class_information,
                    test_groups,
                    test_sample_size,
                    train_groups,
                    test_idx,
                    self.__count_use["test"],
                )

            self.__count_use["train"][train_idx] += 1
            self.__count_use["test"][test_idx] += 1

            assert len(train_idx) == len(
                np.unique(train_idx)
            ), "Duplicate samples used in train set"
            assert len(test_idx) == len(
                np.unique(test_idx)
            ), "Duplicate samples used in test set"

            assert (
                self.__count_use["train"].max() <= self.n_splits
            ), "Train sample used more than n_splits times"
            assert (
                self.__count_use["test"].max() <= self.n_splits
            ), "Test sample used more than n_splits times"

            yield train_idx, test_idx

    def stats(self):
        ic("TRAIN")
        ic("maximum use", self.__count_use["train"].max())
        ic(np.unique(self.__count_use["train"], return_counts=True))
        print()
        ic("TEST")
        ic("maximum use", self.__count_use["test"].max())
        ic(np.unique(self.__count_use["test"], return_counts=True))

    @staticmethod
    def __resolve_group_overlap(
        class_information,
        current_set_groups,
        number_of_samples,
        other_sets_groups,
        result_idx,
        sample_usage,
    ):
        cum_sum = np.cumsum(
            class_information["samples_in_group"][class_information["start_index"] :]
        )
        cum_sum -= number_of_samples

        i = np.argmin(np.abs(cum_sum))
        if cum_sum[i] < 0:
            i += 1

        remaining_groups = class_information["groups"][
            class_information["start_index"] :
        ]
        remaining_samples_in_group = class_information["samples_in_group"][
            class_information["start_index"] :
        ]
        current_groups = np.copy(remaining_groups[: i + 1])

        common_groups = np.intersect1d(current_groups, other_sets_groups)
        if common_groups.size > 0:
            common_group_idx = np.where(np.isin(current_groups, common_groups))[0]

            for group_to_replace_idx, group_to_replace in zip(
                common_group_idx, common_groups
            ):
                new_group_idx = -1
                group_to_replace_samples = remaining_samples_in_group[
                    np.where(remaining_groups == group_to_replace)[0][0]
                ]

                min_use = sample_usage.min()

                new_group = class_information["groups"][new_group_idx]
                current_sample_usage = sample_usage[
                    class_information["sample_indices"][
                        class_information["sample_group"] == new_group
                    ]
                ].max()
                while (
                    new_group in other_sets_groups
                    or new_group in current_groups
                    or class_information["samples_in_group"][new_group_idx]
                    != group_to_replace_samples
                    or current_sample_usage != min_use
                ):
                    new_group_idx -= 1

                    if class_information["extra_samples"] + new_group_idx + 1 < 0:
                        # if all the extra elements with minimal use are checked and none of them is suitable
                        # then start again from the end after increasing the minimal use limit
                        new_group_idx = -1
                        min_use += 1

                    new_group = class_information["groups"][new_group_idx]
                    current_sample_usage = sample_usage[
                        class_information["sample_indices"][
                            class_information["sample_group"] == new_group
                        ]
                    ].max()
                current_groups[group_to_replace_idx] = class_information["groups"][
                    new_group_idx
                ]

        idx = class_information["sample_indices"][
            np.isin(class_information["sample_group"], current_groups)
        ]
        class_information["start_index"] += i + 1
        return np.concatenate([result_idx, idx]), np.concatenate(
            [current_set_groups, current_groups]
        )


if __name__ == "__main__":
    import pandas as pd

    from xai_mam.dataset.metadata import DATASETS

    metadata = pd.read_csv(
        DATASETS["MIAS"].METADATA.FILE, **DATASETS["MIAS"].METADATA.PARAMETERS
    )
    metadata = metadata[~metadata[("normal_vs_benign_vs_malignant", "label")].isna()]

    cv = BalancedGroupKFold(n_splits=10)
    for tr, ts in cv.split(
        metadata,
        y=metadata[("normal_vs_benign_vs_malignant", "label")],
        groups=metadata.index.get_level_values("patient_id"),
    ):
        pass
    cv.stats()
