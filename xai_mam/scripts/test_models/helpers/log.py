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
