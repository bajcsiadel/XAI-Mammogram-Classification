import dataclasses as dc
import inspect
import json
import numpy as np
import os
import subprocess
import torch


def list_of_distances(X, Y):
    return torch.sum(
        (torch.unsqueeze(X, dim=2) - torch.unsqueeze(Y.t(), dim=0)) ** 2, dim=1
    )


def make_one_hot(target, target_one_hot):
    target = target.view(-1, 1)
    target_one_hot.zero_()
    target_one_hot.scatter_(dim=1, index=target, value=1.0)


def makedir(path):
    """
    if path does not exist in the file system, create it
    """
    if not os.path.exists(path):
        os.makedirs(path)


def print_and_write(str, file):
    print(str)
    file.write(str + "\n")


def find_high_activation_crop(activation_map, percentile=95):
    threshold = np.percentile(activation_map, percentile)
    mask = np.ones(activation_map.shape)
    mask[activation_map < threshold] = 0
    lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0
    for i in range(mask.shape[0]):
        if np.amax(mask[i]) > 0.5:
            lower_y = i
            break
    for i in reversed(range(mask.shape[0])):
        if np.amax(mask[i]) > 0.5:
            upper_y = i
            break
    for j in range(mask.shape[1]):
        if np.amax(mask[:, j]) > 0.5:
            lower_x = j
            break
    for j in reversed(range(mask.shape[1])):
        if np.amax(mask[:, j]) > 0.5:
            upper_x = j
            break
    return lower_y, upper_y + 1, lower_x, upper_x + 1


def load_model_parallel(load_model_dir, load_model_name):
    load_model_path = os.path.join(load_model_dir, "model", load_model_name)
    print("load model from " + load_model_path)
    ppnet = torch.load(load_model_path)
    ppnet = ppnet.cuda()
    return torch.nn.DataParallel(ppnet)


def iou(m1, m2):
    intersection = np.logical_and(m1, m2)
    union = np.logical_or(m1, m2)
    return np.sum(intersection) / np.sum(union)


def get_last_commit_hash():
    """
    Get the hash of the last commit
    :return:
    :rtype: str
    """
    process = subprocess.Popen(['git', 'rev-parse', 'HEAD'], shell=False, stdout=subprocess.PIPE)
    return process.communicate()[0].strip()


def get_function_name():
    """
    Get the name of the function that called this function. The first index (1) is the
    function that called this and the second index (3) is the name of that function
    :return:
    :rtype: str
    """
    return inspect.stack()[1][3]


def set_used_images(dataset_config, used_images, target):
    """
    Set the used images for the given dataset config and target
    :param dataset_config:
    :type dataset_config: ProtoPNet.dataset.metadata.DatasetInformation
    :param used_images:
    :type used_images: str
    :param target:
    :type target: str
    """
    assert target in dataset_config.TARGET_TO_VERSION.keys(), "Target does not exist in dataset!"
    versions_key = dataset_config.TARGET_TO_VERSION[target]

    assert used_images in dataset_config.VERSIONS[versions_key].keys(), \
        f"Used images does not exist in dataset for target {target}!"
    dataset_config.USED_IMAGES = dataset_config.VERSIONS[versions_key][used_images]


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if hasattr(o, "to_json"):
            return o.to_json()
        elif dc.is_dataclass(o):
            return dc.asdict(o)
        return super().default(o)
