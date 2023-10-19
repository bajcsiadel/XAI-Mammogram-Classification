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
    :return: str
    """
    process = subprocess.Popen(['git', 'rev-parse', 'HEAD'], shell=False, stdout=subprocess.PIPE)
    return process.communicate()[0].strip()
