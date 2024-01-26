"""

"""
import numpy as np
import pandas as pd
import pipe
from icecream import ic
from sklearn.model_selection import train_test_split


def stratified_grouped_train_test_split(X, y, groups, test_size=0.3, train_size=None, random_state=1234):
    """
    Create a grouped, stratified split of the dataset.

    :param X:
    :param y:
    :param groups:
    :type groups: np.ndarray
    :param test_size: size of the test set. If float between 0 and 1, it is interpreted as a fraction of the dataset.
    If int, it is interpreted as the number of samples.
    :type test_size: float | int
    :param train_size: size of the train set. If float between 0 and 1, it is interpreted as a fraction of the dataset.
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
            train_size = 1. - test_size

    if type(train_size) != type(test_size):
        raise ValueError("train_size and test_size must be of the same type\n"
                         f"{type(train_size) = }\n{type(test_size) = }")

    if type(train_size) is float:
        if not 0 < train_size < 1:
            raise ValueError("train_size must be between 0 and 1")
        if not 0 < test_size < 1:
            raise ValueError("train_size must be between 0 and 1")
        if train_size + test_size != 1:
            raise RuntimeWarning("train_size and test_size do not add up to 1\n"
                                 f"{train_size + test_size} != 1")

        train_size = int(len(X) * train_size)
        test_size = len(X) - train_size

    if train_size + test_size > len(X):
        raise ValueError("train_size + test_size must be smaller than the dataset size\n"
                         f"{train_size + test_size} > {len(X)}")

    if train_size < test_size:
        raise RuntimeWarning("train_size < test_size\n"
                             f"{train_size} < {test_size}")

    if len(X) != len(y) != len(groups):
        raise ValueError("X, y and groups must have the same length")

    # split the indices
    # define case distribution
    cases_distribution = __get_case_distribution(y, groups)
    single_classes = __get_single_groups(cases_distribution)

    cases_distribution.drop(index=single_classes.index, inplace=True)

    train_ratio = train_size / (train_size + test_size)
    test_ratio = 1 - train_ratio
    train_patients, test_patients = train_test_split(cases_distribution.index.to_numpy(),
                                                     test_size=test_ratio, train_size=train_ratio,
                                                     stratify=cases_distribution.values, random_state=random_state)

    train_indices = np.where([patient_id in train_patients for patient_id in groups])[0]
    test_indices = np.where([patient_id in test_patients for patient_id in groups])[0]

    if len(train_indices) > train_size:
        train_indices = np.random.choice(train_indices, train_size, replace=False)
    if len(test_indices) > test_size:
        test_indices = np.random.choice(test_indices, test_size, replace=False)

    # if there are fewer elements in the generated set than needed
    # then add the single classes to the sets
    i = 0
    if len(train_indices) < train_size:
        train_indices, i = __add_single_classes(train_indices, train_size, single_classes, i, groups)
    if len(test_indices) < test_size:
        test_indices, _ = __add_single_classes(test_indices, test_size, single_classes, i, groups)

    return train_indices, test_indices


def __add_single_classes(set_, preferred_set_size, single_classes, index, groups):
    """
    Add the single classes to the sets.
    :param set_: subset of indices
    :type set_: np.ndarray
    :param preferred_set_size:
    :type preferred_set_size: int
    :param single_classes:
    :type single_classes: pd.DataFrame
    :param index:
    :type index: int
    :param groups:
    :return:
    """
    difference = preferred_set_size - len(set_)
    while difference > 0:
        if index == len(single_classes):
            raise UserWarning(f"There are not enough data/groups! "
                              f"Set contains less samples "
                              f"({len(set_)} < {preferred_set_size})")
        cases = np.where(single_classes.iloc[index].name == groups)[0]
        if len(cases) > difference:
            cases = np.random.choice(cases, difference, replace=False)

        difference -= len(cases)
        set_ = np.concatenate((set_, cases))
        index += 1

    return set_, index


def __get_case_distribution(y, groups):
    """
    Get the cases grouped by patient.
    :param y:
    :type y: list | np.ndarray
    :param groups:
    :return: list | np.ndarray
    :rtype: pd.DataFrame
    """
    stacked = sorted(zip(y, groups), key=lambda x: x[0])
    unique_targets = np.unique(y)
    class_to_index_mapping = {
        unique_target: i
        for i, unique_target in enumerate(unique_targets)
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

    import ProtoPNet.util.config_types as conf_typ
    from ProtoPNet.dataset.dataloaders import CustomDataModule

    conf_typ.init_config_store()

    @hydra.main(version_base=None, config_path="../../conf", config_name="main_config")
    def test_split(cfg: conf_typ.Config):

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
        ic(len(train_indices), train_indices)
        ic(len(test_indices), test_indices)

    test_split()
