# MODEL AND DATA LOADING
import argparse
import copy
import os
import re

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
from albumentations.pytorch import ToTensorV2
from ProtoPNet.config.settings import base_architecture, experiment_run
from ProtoPNet.util.helpers import (
    find_high_activation_crop,
    load_model_parallel,
    makedir,
)
from ProtoPNet.util.log import create_logger
from ProtoPNet.util.preprocess import mean, std
from ProtoPNet.util.save import save_image
from torch.autograd import Variable

"""
    Usage: python local_analysis2.py -model='' -imgdir='' -imgclass 1 -gpuid 0
    - model: model name (.pth file)
    - imgdir: path to image
    - imgclass: class of the image
    - gpuid: GPU id
"""
# python post_analysis/local_analysis2.py
#   -model='8_9push0.8089.pth'
#   -imgdir='/bigdata/shared/data/bosch/smaller/L4-12-bits/RoadTypeDetection/
#           LUV_images/LB-XZ_557_20190508_095840_139/
#           LB-XZ_557_20190508_095840_139-2_f001800_fc00383654_609cb4e.png'
#   -imgclass 1
#   -gpuid 2

parser = argparse.ArgumentParser()
parser.add_argument("-gpuid", nargs=1, type=str, default="0")
parser.add_argument("-model", nargs=1, type=str)
parser.add_argument("-imgdir", nargs=1, type=str)
parser.add_argument("-imgclass", nargs=1, type=int, default=-1)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid[0]

experiment_dir = os.path.join(
    os.getcwd(), "saved_models", base_architecture, experiment_run
)
model_dir = os.path.join(experiment_dir, "model")
model_name = args.model[0]
model = ".".join(model_name.split(".")[:-1])
misclass_dir = os.path.join(experiment_dir, model, "misclassification")

test_image_dir = args.imgdir[0]  # from csv file
test_image_label = args.imgclass[0]

save_analysis_path = os.path.join(experiment_dir, model, "local_analysis")
makedir(save_analysis_path)

log, logclose = create_logger(
    log_filename=os.path.join(save_analysis_path, "local_analysis.log")
)

epoch_number_str = re.search(r"\d+", model_name).group(0)
start_epoch_number = int(epoch_number_str)

log("model base architecture: " + base_architecture)
log("experiment run: " + experiment_run)

ppnet_multi = load_model_parallel(model_dir, model_name)

img_shape = ppnet_multi.module.img_shape
prototype_shape = ppnet_multi.module.prototype_shape
max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

class_specific = True

normalize = transforms.Normalize(mean=mean, std=std)

# SANITY CHECK
# confirm prototype class identity
load_img_dir = os.path.join(experiment_dir, "img")

prototype_info = np.load(
    os.path.join(
        load_img_dir,
        "epoch-" + epoch_number_str,
        "bb" + epoch_number_str + ".npy",
    )
)
prototype_img_identity = prototype_info[:, -1]

log(
    "Prototypes are chosen from "
    + str(len(set(prototype_img_identity)))
    + " number of classes."
)
log("Their class identities are: " + str(prototype_img_identity))

# confirm prototype connects most strongly to its own class
prototype_max_connection = torch.argmax(
    ppnet_multi.module.last_layer.weight, dim=0
)
prototype_max_connection = prototype_max_connection.cpu().numpy()
if (
    np.sum(prototype_max_connection == prototype_img_identity)
    == ppnet_multi.module.num_prototypes
):
    log("All prototypes connect most strongly to their respective classes.")
else:
    log(
        "WARNING: Not all prototypes connect most strongly to their respective classes."
    )


# HELPER FUNCTIONS FOR PLOTTING
def save_preprocessed_img(fname, preprocessed_imgs, index=0):
    img_copy = copy.deepcopy(preprocessed_imgs[index : index + 1])
    undo_preprocessed_img = img_copy
    print("image index {0} in batch".format(index))
    undo_preprocessed_img = undo_preprocessed_img[0]
    undo_preprocessed_img = undo_preprocessed_img.detach().cpu().numpy()
    undo_preprocessed_img = np.transpose(undo_preprocessed_img, [1, 2, 0])

    save_image(fname, undo_preprocessed_img)
    return undo_preprocessed_img


def save_prototype(fname, epoch, index):
    p_img = plt.imread(
        os.path.join(
            load_img_dir,
            "epoch-" + str(epoch),
            "prototype-img" + str(index) + ".png",
        )
    )
    # plt.axis('off')
    plt.imsave(fname, p_img)


def save_prototype_self_activation(fname, epoch, index):
    p_img = plt.imread(
        os.path.join(
            load_img_dir,
            "epoch-" + str(epoch),
            "prototype-img-original_with_self_act" + str(index) + ".png",
        )
    )
    # plt.axis('off')
    plt.imsave(fname, p_img)


def save_prototype_original_img_with_bbox(
    fname,
    epoch,
    index,
    bbox_height_start,
    bbox_height_end,
    bbox_width_start,
    bbox_width_end,
    color=(0, 255, 255),
):
    p_img_bgr = cv2.imread(
        os.path.join(
            load_img_dir,
            "epoch-" + str(epoch),
            "prototype-img-original" + str(index) + ".png",
        )
    )
    cv2.rectangle(
        p_img_bgr,
        (bbox_width_start, bbox_height_start),
        (bbox_width_end - 1, bbox_height_end - 1),
        color,
        thickness=2,
    )
    p_img_rgb = p_img_bgr[..., ::-1]
    p_img_rgb = np.float32(p_img_rgb) / 255
    # plt.imshow(p_img_rgb)
    # plt.axis('off')
    plt.imsave(fname, p_img_rgb)


def imsave_with_bbox(
    fname,
    img_rgb,
    bbox_height_start,
    bbox_height_end,
    bbox_width_start,
    bbox_width_end,
    color=(0, 255, 255),
):
    img_bgr_uint8 = cv2.cvtColor(np.uint8(255 * img_rgb), cv2.COLOR_RGB2BGR)
    cv2.rectangle(
        img_bgr_uint8,
        (bbox_width_start, bbox_height_start),
        (bbox_width_end - 1, bbox_height_end - 1),
        color,
        thickness=2,
    )
    img_rgb_uint8 = img_bgr_uint8[..., ::-1]
    img_rgb_float = np.float32(img_rgb_uint8) / 255
    # plt.imshow(img_rgb_float)
    # plt.axis('off')
    plt.imsave(fname, img_rgb_float)


# load the test image and forward it through the network
preprocess = A.Compose(
    [
        A.ToFloat(max_value=4096),  # 12-bits
        A.augmentations.geometric.resize.Resize(
            height=img_shape[0], width=img_shape[1]
        ),
        ToTensorV2(),
    ]
)

img = cv2.imread(test_image_dir, cv2.IMREAD_UNCHANGED)
img_tensor = preprocess(image=img)["image"]
img_variable = Variable(img_tensor.unsqueeze(0))

images_test = img_variable.cuda()
labels_test = torch.tensor([test_image_label])

logits, distance_dict = ppnet_multi(images_test)
conv_output, distances = ppnet_multi.module.push_forward(images_test)
prototype_activations = ppnet_multi.module.distance_2_similarity(
    distance_dict["min_distances"]
)
prototype_activation_patterns = ppnet_multi.module.distance_2_similarity(
    distances
)
if ppnet_multi.module.prototype_activation_function == "linear":
    prototype_activations = prototype_activations + max_dist
    prototype_activation_patterns = prototype_activation_patterns + max_dist

tables = []
for i in range(logits.size(0)):
    tables.append(
        (torch.argmax(logits, dim=1)[i].item(), labels_test[i].item())
    )
    log(str(i) + " " + str(tables[-1]))

idx = 0
predicted_cls = tables[idx][0]
correct_cls = tables[idx][1]
log("Predicted: " + str(predicted_cls))
log("Actual: " + str(correct_cls))
original_img = save_preprocessed_img(
    os.path.join(save_analysis_path, "original_img.png"), images_test, idx
)

# MOST ACTIVATED (NEAREST) 10 PROTOTYPES OF THIS IMAGE
makedir(os.path.join(save_analysis_path, "most_activated_prototypes"))

log("Most activated 10 prototypes of this image:")
array_act, sorted_indices_act = torch.sort(prototype_activations[idx])
for i in range(1, 11):
    log("top {0} activated prototype for this image:".format(i))
    save_prototype(
        os.path.join(
            save_analysis_path,
            "most_activated_prototypes",
            "top-%d_activated_prototype.png" % i,
        ),
        start_epoch_number,
        sorted_indices_act[-i].item(),
    )
    save_prototype_original_img_with_bbox(
        fname=os.path.join(
            save_analysis_path,
            "most_activated_prototypes",
            "top-%d_activated_prototype_in_original_pimg.png" % i,
        ),
        epoch=start_epoch_number,
        index=sorted_indices_act[-i].item(),
        bbox_height_start=prototype_info[sorted_indices_act[-i].item()][1],
        bbox_height_end=prototype_info[sorted_indices_act[-i].item()][2],
        bbox_width_start=prototype_info[sorted_indices_act[-i].item()][3],
        bbox_width_end=prototype_info[sorted_indices_act[-i].item()][4],
        color=(0, 255, 255),
    )
    save_prototype_self_activation(
        os.path.join(
            save_analysis_path,
            "most_activated_prototypes",
            "top-%d_activated_prototype_self_act.png" % i,
        ),
        start_epoch_number,
        sorted_indices_act[-i].item(),
    )
    log("prototype index: {0}".format(sorted_indices_act[-i].item()))
    log(
        "prototype class identity: {0}".format(
            prototype_img_identity[sorted_indices_act[-i].item()]
        )
    )
    if (
        prototype_max_connection[sorted_indices_act[-i].item()]
        != prototype_img_identity[sorted_indices_act[-i].item()]
    ):
        log(
            "prototype connection identity: {0}".format(
                prototype_max_connection[sorted_indices_act[-i].item()]
            )
        )
    log("activation value (similarity score): {0}".format(array_act[-i]))
    log(
        "last layer connection with predicted class: {0}".format(
            ppnet_multi.module.last_layer.weight[predicted_cls][
                sorted_indices_act[-i].item()
            ]
        )
    )

    activation_pattern = (
        prototype_activation_patterns[idx][sorted_indices_act[-i].item()]
        .detach()
        .cpu()
        .numpy()
    )
    upsampled_activation_pattern = cv2.resize(
        activation_pattern,
        dsize=(img_shape[1], img_shape[0]),
        interpolation=cv2.INTER_CUBIC,
    )

    # show the most highly activated patch of the image by this prototype
    high_act_patch_indices = find_high_activation_crop(
        upsampled_activation_pattern
    )
    high_act_patch = original_img[
        high_act_patch_indices[0] : high_act_patch_indices[1],
        high_act_patch_indices[2] : high_act_patch_indices[3],
        :,
    ]
    log("most highly activated patch of the chosen image by this prototype:")
    # plt.axis('off')
    save_image(
        os.path.join(
            save_analysis_path,
            "most_activated_prototypes",
            "most_highly_activated_patch_by_top-%d_prototype.png" % i,
        ),
        high_act_patch,
    )
    log(
        "most highly activated patch by this prototype shown in the original image:"
    )
    imsave_with_bbox(
        fname=os.path.join(
            save_analysis_path,
            "most_activated_prototypes",
            "most_highly_activated_patch_in_original_img_by_top-%d_prototype.png"
            % i,
        ),
        img_rgb=original_img,
        bbox_height_start=high_act_patch_indices[0],
        bbox_height_end=high_act_patch_indices[1],
        bbox_width_start=high_act_patch_indices[2],
        bbox_width_end=high_act_patch_indices[3],
        color=(0, 255, 255),
    )

    # show the image overlayed with prototype activation map
    rescaled_activation_pattern = upsampled_activation_pattern - np.amin(
        upsampled_activation_pattern
    )
    rescaled_activation_pattern = rescaled_activation_pattern / np.amax(
        rescaled_activation_pattern
    )
    heatmap = cv2.applyColorMap(
        np.uint8(255 * rescaled_activation_pattern), cv2.COLORMAP_JET
    )
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]
    overlayed_img = 0.5 * original_img + 0.3 * heatmap
    log("prototype activation map of the chosen image:")
    # plt.axis('off')
    plt.imsave(
        os.path.join(
            save_analysis_path,
            "most_activated_prototypes",
            "prototype_activation_map_by_top-%d_prototype.png" % i,
        ),
        overlayed_img,
    )
    log("--------------------------------------------------------------")

# PROTOTYPES FROM TOP-k CLASSES
k = 1
log("Prototypes from top-%d classes:" % k)
topk_logits, topk_classes = torch.topk(logits[idx], k=k)
for i, c in enumerate(topk_classes.detach().cpu().numpy()):
    makedir(
        os.path.join(save_analysis_path, "top-%d_class_prototypes" % (i + 1))
    )

    log("top %d predicted class: %d" % (i + 1, c))
    log("logit of the class: %f" % topk_logits[i])
    class_prototype_indices = np.nonzero(
        ppnet_multi.module.prototype_class_identity.detach()
        .cpu()
        .numpy()[:, c]
    )[0]
    class_prototype_activations = prototype_activations[idx][
        class_prototype_indices
    ]
    _, sorted_indices_cls_act = torch.sort(class_prototype_activations)

    prototype_cnt = 1
    for j in reversed(sorted_indices_cls_act.detach().cpu().numpy()):
        prototype_index = class_prototype_indices[j]
        save_prototype(
            os.path.join(
                save_analysis_path,
                "top-%d_class_prototypes" % (i + 1),
                "top-%d_activated_prototype.png" % prototype_cnt,
            ),
            start_epoch_number,
            prototype_index,
        )
        save_prototype_original_img_with_bbox(
            fname=os.path.join(
                save_analysis_path,
                "top-%d_class_prototypes" % (i + 1),
                "top-%d_activated_prototype_in_original_pimg.png"
                % prototype_cnt,
            ),
            epoch=start_epoch_number,
            index=prototype_index,
            bbox_height_start=prototype_info[prototype_index][1],
            bbox_height_end=prototype_info[prototype_index][2],
            bbox_width_start=prototype_info[prototype_index][3],
            bbox_width_end=prototype_info[prototype_index][4],
            color=(0, 255, 255),
        )
        save_prototype_self_activation(
            os.path.join(
                save_analysis_path,
                "top-%d_class_prototypes" % (i + 1),
                "top-%d_activated_prototype_self_act.png" % prototype_cnt,
            ),
            start_epoch_number,
            prototype_index,
        )
        log("prototype index: {0}".format(prototype_index))
        log(
            "prototype class identity: {0}".format(
                prototype_img_identity[prototype_index]
            )
        )
        if (
            prototype_max_connection[prototype_index]
            != prototype_img_identity[prototype_index]
        ):
            log(
                "prototype connection identity: {0}".format(
                    prototype_max_connection[prototype_index]
                )
            )
        log(
            "activation value (similarity score): {0}".format(
                prototype_activations[idx][prototype_index]
            )
        )
        log(
            "last layer connection: {0}".format(
                ppnet_multi.module.last_layer.weight[c][prototype_index]
            )
        )

        activation_pattern = (
            prototype_activation_patterns[idx][prototype_index]
            .detach()
            .cpu()
            .numpy()
        )
        upsampled_activation_pattern = cv2.resize(
            activation_pattern,
            dsize=(img_shape[1], img_shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )

        # show the most highly activated patch of the image by this prototype
        high_act_patch_indices = find_high_activation_crop(
            upsampled_activation_pattern
        )
        high_act_patch = original_img[
            high_act_patch_indices[0] : high_act_patch_indices[1],
            high_act_patch_indices[2] : high_act_patch_indices[3],
            :,
        ]
        log(
            "most highly activated patch of the chosen image by this prototype:"
        )
        # plt.axis('off')
        save_image(
            os.path.join(
                save_analysis_path,
                "top-%d_class_prototypes" % (i + 1),
                "most_highly_activated_patch_by_top-%d_prototype.png"
                % prototype_cnt,
            ),
            high_act_patch,
        )
        log(
            "most highly activated patch by this prototype shown in the original image:"
        )
        imsave_with_bbox(
            fname=os.path.join(
                save_analysis_path,
                "top-%d_class_prototypes" % (i + 1),
                "most_highly_activated_patch_in_original_img_by_top-%d_prototype.png"
                % prototype_cnt,
            ),
            img_rgb=original_img,
            bbox_height_start=high_act_patch_indices[0],
            bbox_height_end=high_act_patch_indices[1],
            bbox_width_start=high_act_patch_indices[2],
            bbox_width_end=high_act_patch_indices[3],
            color=(0, 255, 255),
        )

        # show the image overlayed with prototype activation map
        rescaled_activation_pattern = upsampled_activation_pattern - np.amin(
            upsampled_activation_pattern
        )
        rescaled_activation_pattern = rescaled_activation_pattern / np.amax(
            rescaled_activation_pattern
        )
        heatmap = cv2.applyColorMap(
            np.uint8(255 * rescaled_activation_pattern), cv2.COLORMAP_JET
        )
        heatmap = np.float32(heatmap) / 255
        heatmap = heatmap[..., ::-1]
        overlayed_img = 0.5 * original_img + 0.3 * heatmap
        log("prototype activation map of the chosen image:")
        # plt.axis('off')
        plt.imsave(
            os.path.join(
                save_analysis_path,
                "top-%d_class_prototypes" % (i + 1),
                "prototype_activation_map_by_top-%d_prototype.png"
                % prototype_cnt,
            ),
            overlayed_img,
        )
        log("--------------------------------------------------------------")
        prototype_cnt += 1
    log("***************************************************************")

if predicted_cls == correct_cls:
    log("Prediction is correct.")
else:
    log("Prediction is wrong.")

logclose()
