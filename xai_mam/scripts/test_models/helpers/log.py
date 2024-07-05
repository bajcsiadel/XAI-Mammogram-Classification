import numpy as np
import pandas as pd

from xai_mam.utils.log import ScriptLogger


def print_set_information(
    logger: ScriptLogger, set_name: str, patient_ids: np.ndarray, labels: np.ndarray
):
    """
    Print the information about the given set.

    :param logger: logger to be used
    :param set_name: name of the set
    :param patient_ids: patient IDs
    :param labels: labels of the patients
    """
    logger.info(f"{set_name}")
    logger.increase_indent()

    logger.info(f"Number of patients: {len(patient_ids)}")
    logger.info(f"Number of images: {len(labels)}")
    logger.info(f"IDs: {patient_ids}")
    distribution = pd.DataFrame(np.unique(labels, return_counts=True)).T
    distribution.columns = ["label", "count"]
    distribution["perc"] = distribution["count"] / distribution["count"].sum()
    logger.info(
        distribution.to_string(index=False, formatters={"perc": "{:3.2%}".format})
    )

    logger.decrease_indent()


def print_set_overlap(logger: ScriptLogger, *arrays: tuple[str, np.ndarray]):
    """
    Print the intersection of each pair of sets in the given list of arrays.

    :param logger: logger object
    :param arrays: list of tuples with the first element as the name of
        the set and the second element as the set
    """
    from itertools import combinations
    for set_1, set_2 in combinations(arrays, 2):
        logger.info(f"{set_1[0]} ∩ {set_2[0]}: {set(set_1[1]) & set(set_2[1])}")
