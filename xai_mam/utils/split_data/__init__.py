"""

"""
import warnings

import numpy as np
import pandas as pd
import pipe
from icecream import ic
from sklearn.model_selection import train_test_split


def stratified_grouped_train_test_split(
    X, y, groups, test_size=0.3, train_size=None, random_state=1234
):
    """
    Create a grouped, stratified split of the dataset, where each group
    has the same class distribution as the original dataset.

    :param X:
    :param y:
    :param groups:
    :type groups: np.ndarray
    :param test_size: size of the test set. If float between 0 and 1,
    it is interpreted as a fraction of the dataset.
    If int, it is interpreted as the number of samples.
    :type test_size: float | int
    :param train_size: size of the train set. If float between 0 and 1,
    it is interpreted as a fraction of the dataset.
    If int, it is interpreted as the number of samples.
    :type train_size: float | int | None
    :param random_state:
    :return:
    """
    # validations
    if train_size is None:
        if type(test_size) is int:
            train_size = len(X) - test_size
        else:
            train_size = 1.0 - test_size

    if type(train_size) is not type(test_size):
        raise ValueError(
            "train_size and test_size must be of the same type\n"
            f"{type(train_size) = }\n{type(test_size) = }"
        )

    if type(train_size) is float:
        if not 0 < train_size < 1:
            raise ValueError("train_size must be between 0 and 1")
        if not 0 < test_size < 1:
            raise ValueError("train_size must be between 0 and 1")
        if train_size + test_size != 1:
            raise RuntimeWarning(
                "train_size and test_size do not add up to 1\n"
                f"{train_size + test_size} != 1"
            )

        train_size = int(len(X) * train_size)
        test_size = len(X) - train_size

    if train_size + test_size > len(X):
        raise ValueError(
            "train_size + test_size must be smaller than the dataset size\n"
            f"{train_size + test_size} > {len(X)}"
        )

    if train_size < test_size:
        raise RuntimeWarning("train_size < test_size\n" f"{train_size} < {test_size}")

    if len(X) != len(y) != len(groups):
        raise ValueError("X, y and groups must have the same length")

    # split the indices
    # define case distribution
    cases_distribution = __get_case_distribution(y, groups)
    group_case_count = cases_distribution.sum(axis=1)
    single_classes = __get_single_groups(cases_distribution)

    cases_distribution.drop(index=single_classes.index, inplace=True)

    train_ratio = train_size / (train_size + test_size)
    # removing extra decimals. Sometimes 1.0 - 0.7 = 0.30000000000000004
    test_ratio = np.floor((1 - train_ratio) * 100) / 100

    label_count = np.vstack(np.unique(y, return_counts=True)).T
    original_ratio = label_count[:, 1] / np.sum(label_count[:, 1])

    train_patients, test_patients = train_test_split(
        cases_distribution.index.to_numpy(),
        test_size=test_ratio,
        train_size=train_ratio,
        stratify=cases_distribution.values,
        random_state=random_state,
    )

    train_indices = np.where([patient_id in train_patients for patient_id in groups])[0]
    test_indices = np.where([patient_id in test_patients for patient_id in groups])[0]

    if (
        abs(len(train_indices) - train_size) < group_case_count.min()
        or abs(len(test_indices) - test_size) < group_case_count.min()
    ):
        # we reached the best possible split following a grouped approach
        # we cannot reach the desired set sizes without splitting a group
        return train_indices, test_indices

    if len(train_indices) > train_size:
        train_indices, single_classes = __discard_indices(
            train_indices,
            train_size,
            y,
            groups,
            cases_distribution,
            group_case_count,
            single_classes,
            original_ratio,
        )
    if len(test_indices) > test_size:
        test_indices, single_classes = __discard_indices(
            test_indices,
            test_size,
            y,
            groups,
            cases_distribution,
            group_case_count,
            single_classes,
            original_ratio,
        )

    # if there are fewer elements in the generated set than needed
    # then add the single classes to the sets
    if len(train_indices) < train_size:
        train_indices, single_classes = __add_single_classes(
            train_indices, train_size, single_classes, groups
        )
    if len(test_indices) < test_size:
        test_indices, _ = __add_single_classes(
            test_indices, test_size, single_classes, groups
        )

    return train_indices, test_indices


def __discard_indices(
    selected_indices,
    preferred_set_size,
    y,
    groups,
    cases_distribution,
    group_case_count,
    single_classes,
    original_ratio,
):
    """
    Remove indices from the dataset by keeping the class distribution.

    :param selected_indices: subset of indices
    :type selected_indices: numpy.ndarray
    :param preferred_set_size:
    :type preferred_set_size: int
    :param y: targets of each patient
    :type y: numpy.ndarray
    :param groups: groups of the patients
    :type groups: numpy.ndarray
    :param cases_distribution: distribution of the cases
    :type cases_distribution: pandas.DataFrame
    :param group_case_count: count of cases per group
    :type group_case_count: pandas.Series
    :param single_classes: single classes (not used groups)
    :type single_classes: pandas.DataFrame
    :param original_ratio: original ratio of the classes in the dataset
    :type original_ratio: numpy.ndarray
    :return:
    :rtype: tuple[numpy.ndarray, pandas.DataFrame]
    """
    # filter the groups that are in the selected set
    groups_in_set = np.unique(groups[selected_indices])
    group_case_count = group_case_count.loc[groups_in_set]
    cases_distribution = cases_distribution.loc[groups_in_set]

    size_difference = len(selected_indices) - preferred_set_size
    while size_difference != 0:
        current_counts = np.unique(y[selected_indices], return_counts=True)[1]
        possible_groups_to_remove = (group_case_count - size_difference).abs()
        # both the count difference and the total count difference are
        # negative resulting in a positive ratio value
        new_ratio = (cases_distribution - current_counts) / (
            group_case_count.values - len(selected_indices)
        )[:, np.newaxis]
        ratio_difference = (new_ratio - original_ratio).abs()

        group_data = pd.DataFrame(
            index=possible_groups_to_remove.index, columns=["count", "ratio"]
        )
        group_data["count"] = possible_groups_to_remove
        group_data["ratio"] = ratio_difference.sum(axis=1)
        # sort the values first by count and then by ratio
        group_data = group_data.sort_values(by=["ratio", "count"])

        remove_index = group_data.index[0]
        patient_indices = np.where(remove_index == groups)[0]
        # remove the patient indices corresponding to the group
        if len(patient_indices) > size_difference:
            patient_indices = np.random.choice(
                patient_indices, size_difference, replace=False
            )
        else:
            # if the group is entirely removed, then add it to the single classes
            single_classes.loc[remove_index] = cases_distribution.loc[remove_index]
            cases_distribution.drop(index=remove_index, inplace=True)
            group_case_count.drop(index=remove_index, inplace=True)

        selected_indices = np.setdiff1d(selected_indices, patient_indices)

        size_difference = len(selected_indices) - preferred_set_size
    return selected_indices, single_classes


def __add_single_classes(set_, preferred_set_size, single_classes, groups):
    """
    Add the single classes to the sets.
    :param set_: subset of indices
    :type set_: numpy.ndarray
    :param preferred_set_size:
    :type preferred_set_size: int
    :param single_classes:
    :type single_classes: pandas.DataFrame
    :param index:
    :type index: int
    :param groups:
    :type groups: numpy.ndarray
    :return:
    :rtype: tuple[numpy.ndarray, pandas.DataFrame]
    """
    difference = preferred_set_size - len(set_)
    while difference > 0:
        closest = (single_classes.sum(axis=1) - difference).abs()
        closest = closest.sort_values()
        if len(single_classes) == 0:
            warnings.warn(
                f"There are not enough data/groups! "
                f"Set contains less samples "
                f"({len(set_)} < {preferred_set_size})",
                stacklevel=2,
            )
            break
        index = closest.index[0]
        cases = np.where(single_classes.loc[index].name == groups)[0]
        if len(cases) > difference:
            cases = np.random.choice(cases, difference, replace=False)

        difference -= len(cases)
        single_classes.drop(index, inplace=True)
        set_ = np.concatenate((set_, cases))

    return set_, single_classes


def __get_case_distribution(y, groups):
    """
    Get the cases grouped by patient.
    :param y:
    :type y: list | np.ndarray
    :param groups:
    :return: list | np.ndarray
    :rtype: pd.DataFrame
    """
    stacked = sorted(zip(y, groups, strict=True), key=lambda x: x[0])
    unique_targets = np.unique(y)
    class_to_index_mapping = {
        unique_target: i for i, unique_target in enumerate(unique_targets)
    }
    patients_cases = []

    grouped = stacked | pipe.sort(lambda x: x[1]) | pipe.groupby(lambda x: x[1])

    for _, cases_in_group in grouped:
        current_patient_cases = [0] * len(class_to_index_mapping)
        for case, _ in cases_in_group:
            current_patient_cases[class_to_index_mapping[case]] += 1
        patients_cases.append(current_patient_cases)

    return pd.DataFrame(patients_cases, columns=unique_targets, index=np.unique(groups))


def __get_single_groups(groups):
    """
    Get the groups with a single instance.
    :param groups:
    :type groups: pd.DataFrame
    :return:
    :rtype: pd.DataFrame
    """
    _, indices, count = np.unique(groups, axis=0, return_counts=True, return_index=True)
    return groups.iloc[indices[count == 1]]


if __name__ == "__main__":
    import hydra
    from omegaconf import omegaconf

    from xai_mam.dataset.dataloaders import CustomDataModule
    from xai_mam.utils.config.script_main import Config

    Config.init_store()

    @hydra.main(version_base=None, config_path="../../conf", config_name="main_config")
    def test_split(cfg: Config):
        cfg = omegaconf.OmegaConf.to_object(cfg)

        data_module: CustomDataModule = hydra.utils.instantiate(cfg.data.datamodule)
        train_indices, test_indices = stratified_grouped_train_test_split(
            data_module.train_data.metadata,
            data_module.train_data.targets,
            data_module.train_data.groups,
            train_size=100,
            test_size=25,
            random_state=cfg.seed,
        )
        ic(len(data_module.train_data.metadata))
        ic(len(train_indices), train_indices)
        ic(len(test_indices), test_indices)

        sizes = pd.DataFrame(
            [
                [len(data_module.train_data.metadata)],
                [len(train_indices)],
                [len(test_indices)],
                [len(train_indices) + len(test_indices)],
            ],
            columns=["count"],
            index=["original", "train", "test", "all split"],
        )
        ic(sizes)

        for indices in [
            range(len(data_module.train_data.metadata)),
            train_indices,
            test_indices,
        ]:
            distribution = pd.DataFrame(columns=["count", "perc"])
            targets = data_module.train_data.targets[indices]
            classes = np.unique(targets)
            for cls in classes:
                count = np.sum(targets == cls)
                distribution.loc[cls] = [count, count / len(indices)]
            distribution.loc["TOTAL"] = distribution.sum().values
            distribution["count"] = distribution["count"].astype("int")

            ic(distribution)

    test_split()
