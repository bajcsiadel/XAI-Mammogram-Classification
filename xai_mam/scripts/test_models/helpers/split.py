import numpy as np
from sklearn.model_selection import train_test_split


def _select_elements(indices: np.ndarray, *dicts: dict | list | np.ndarray) -> list:
    """
    Select the elements defined by the indices from the dictionaries/arrays.

    :param indices: indices to be selected from the dictionaries/arrays
    :param dicts: dictionaries/arrays to select the elements from
    :return: list containing a sublist for each dictionary/array
        with the selected elements
    """
    results = []
    for current_dict in dicts:
        results.append([])
        for i in indices:
            if type(current_dict[i]) is dict:
                values = current_dict[i].values()
            else:
                values = [current_dict[i]]
            results[-1].extend(list(values))
        results[-1] = np.array(results[-1])
    return results


def patient_split_data(
    patient_ids: np.ndarray,
    labels: np.ndarray,
    *data: dict | list | np.ndarray,
    test_size: float = 0.15,
    random_state: int = 2021,
    shuffle: bool = True,
) -> tuple:
    """
    Split the data into two mutually exclusive sets based on the patient IDs.

    :param patient_ids: patient IDs
    :param labels: labels of the patients
    :param data: dictionaries/arrays to select the elements from
    :param test_size: proportion of the data to include in the test split.
        Defaults to ``0.15``.
    :param random_state: seed used by the random number generator. Defaults to ``2021``.
    :param shuffle: whether to shuffle the data before splitting. Defaults to ``True``.
    """
    [unique_patients, unique_labels] = np.unique(
        np.vstack((patient_ids, labels)), axis=1
    )
    train_indices, test_indices = train_test_split(
        unique_patients,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=unique_labels,
    )
    return (
        *_select_elements(train_indices, *data),
        *_select_elements(test_indices, *data),
        train_indices,
        test_indices,
        np.array(
            [np.argwhere(patient_ids == patient_id) for patient_id in train_indices]
        ).flatten(),
    )


def random_split_data(
    patient_ids: np.ndarray,
    indices: np.ndarray,
    *data: dict | list | np.ndarray,
    test_size: float = 0.15,
    random_state: int = 2021,
    shuffle: bool = True,
) -> tuple:
    """
    Split the data randomly into two sets.

    :param patient_ids: all patient IDs
    :param indices: indices to be split
    :param data: dictionaries/arrays to select the elements from
    :param test_size: proportion of the data to include in the test split.
        Defaults to ``0.15``.
    :param random_state: seed used by the random number generator. Defaults to ``2021``.
    :param shuffle: whether to shuffle the data before splitting. Defaults to ``True``.
    """
    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
    )
    return (
        *_select_elements(train_indices, *data),
        *_select_elements(test_indices, *data),
        patient_ids[train_indices],
        patient_ids[test_indices],
        train_indices,
    )
