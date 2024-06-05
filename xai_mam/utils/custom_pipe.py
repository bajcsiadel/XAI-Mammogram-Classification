import numpy as np
from pipe import *


@Pipe
def to_list(iterable):
    """
    Convert an iterable to a list (using with pipe)

    :param iterable:
    :return:
    :rtype: list
    """
    return list(iterable)


@Pipe
def to_numpy(iterable):
    """
    Convert an iterable to a list (using with pipe)

    :param iterable:
    :return:
    :rtype: list
    """
    return np.array(list(iterable))
