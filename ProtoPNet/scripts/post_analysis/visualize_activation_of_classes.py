# MODEL AND DATA LOADING
import argparse
import copy
import json
import os
import re

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from albumentations.pytorch import ToTensorV2
from ProtoPNet.config.settings import (
    base_architecture,
    experiment_run,
    num_classes,
    num_prototypes_per_class,
)
from ProtoPNet.util.helpers import (
    find_high_activation_crop,
    load_model_parallel,
    makedir,
)
from ProtoPNet.util.log import create_logger
from ProtoPNet.util.save import save_image
from torch.autograd import Variable

'''
    Usage: python visualize_activation_of_classes.py -model='' -examplefile='' -gpuid 0 
        OR python visualize_activation_of_classes.py -model='' -imgdir='' -imgclass 1 -gpuid 0 
    - model: model name (.pth file)
    - examplefile: file containing image paths
    - imgdir: image path
    - imgclass: original class of the image
    - gpuid: GPU id    
    Visualizes the activation of prototypes per each class on given image/images (examplefile)
'''


# python post_analysis/visualize_activation_of_classes.py -model='8_9push0.8089.pth' -examplefile='./saved_models/resnet18/BOSCH_211/8_9push0.8089/examples.csv' -gpuid 2


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
        ),
        cv2.IMREAD_UNCHANGED,
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


def classification_type(sample: dict):
    if sample["true_class"] == sample["predicted_class_avg"]:
        if sample["true_class"] == sample["predicted_class_max"]:
            return "both_correct"
        else:
            return "avg_correct"
    else:
        if sample["true_class"] == sample["predicted_class_max"]:
            return "avg_miss"
        else:
            return "both_miss"


parser = argparse.ArgumentParser()
parser.add_argument("-gpuid", type=str, default="0")
parser.add_argument("-model", type=str)
parser.add_argument("-imgdir", type=str)
parser.add_argument("-examplefile", type=str)
parser.add_argument("-imgclass", type=int, default=-1)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid

experiment_dir = os.path.join(
    os.getcwd(), "saved_models", base_architecture, experiment_run
)
model_dir = os.path.join(experiment_dir, "model")
model_name = args.model
model = ".".join(model_name.split(".")[:-1])
misclass_dir = os.path.join(experiment_dir, model, "misclassification")

heatmap_dir = os.path.join(experiment_dir, model, "heatmaps")
makedir(heatmap_dir)

log, logclose = create_logger(
    log_filename=os.path.join(heatmap_dir, "local_analysis.log")
)

if args.imgdir is None and args.examplefile is None:
    log("ERROR! Image not specified!")
    raise FileNotFoundError("Image not specified!")

images = []
if args.examplefile is not None:
    examples = pd.read_csv(args.examplefile, header=0)
    images = [value for value in examples.T.to_dict().values()]

if args.imgdir is not None:
    images.append({"image_path": args.imgdir, "true_label": args.imgclass})

epoch_number_str = re.search(r"\d+", model_name).group(0)
start_epoch_number = int(epoch_number_str)

log("model base architecture: " + base_architecture)
log("experiment run: " + experiment_run)

ppnet_multi = load_model_parallel(model_dir, model_name)

img_shape = ppnet_multi.module.img_shape
prototype_shape = ppnet_multi.module.prototype_shape
max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

class_specific = True

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

for example in images:
    test_image_path = example["image_path"]
    test_image_label = example["true_class"]

    img_name, _ = os.path.splitext(os.path.basename(test_image_path))
    save_analysis_path = os.path.join(
        heatmap_dir, classification_type(example), img_name
    )
    makedir(save_analysis_path)

    img = cv2.imread(test_image_path, cv2.IMREAD_UNCHANGED)
    img_tensor = preprocess(image=img)["image"]
    img_variable = Variable(img_tensor.unsqueeze(0))

    images_test = img_variable.cuda()
    labels_test = torch.tensor([test_image_label])

    logits, distance_dict = ppnet_multi(images_test)
    prototype_activations = ppnet_multi.module.distance_2_similarity(
        distance_dict["min_distances"]
    )
    prototype_activation_patterns = ppnet_multi.module.distance_2_similarity(
        distance_dict["distances"]
    )
    if ppnet_multi.module.prototype_activation_function == "linear":
        prototype_activations = prototype_activations + max_dist
        prototype_activation_patterns = (
            prototype_activation_patterns + max_dist
        )

    tables = []
    for i in range(logits.size(0)):
        tables.append(
            (torch.argmax(logits, dim=1)[i].item(), labels_test[i].item())
        )
        log(str(i) + " " + str(tables[-1]))

    predicted_cls = tables[0][0]
    correct_cls = tables[0][1]
    log("Predicted: " + str(predicted_cls))
    log("Actual: " + str(correct_cls))
    original_img = save_preprocessed_img(
        os.path.join(
            save_analysis_path, f"original_img_{test_image_label}.png"
        ),
        images_test,
    )
    json.dump(
        example, open(os.path.join(save_analysis_path, "details.json"), "w")
    )
    for i in range(num_classes):
        max_activation_pattern = np.zeros((img_shape[0], img_shape[1]))
        for j in range(num_prototypes_per_class):
            prot_idx = i * num_prototypes_per_class + j
            log(f"prototype index: {prot_idx}")
            log(
                f"prototype class identity: {prototype_img_identity[prot_idx]}"
            )
            if (
                prototype_max_connection[prot_idx]
                != prototype_img_identity[prot_idx]
            ):
                proto_conn_id = prototype_max_connection[prot_idx]
                log(f"prototype connection identity: {proto_conn_id}")
            act_value = prototype_activations[0][prot_idx]
            log(f"activation value (similarity score): {act_value}")
            connection = ppnet_multi.module.last_layer.weight[predicted_cls][
                prot_idx
            ]
            log(f"last layer connection with predicted class: {connection}")

            activation_pattern = (
                prototype_activation_patterns[0][prot_idx]
                .detach()
                .cpu()
                .numpy()
                * ppnet_multi.module.last_layer.weight[predicted_cls][prot_idx]
                .detach()
                .cpu()
                .numpy()
            )

            upsampled_activation_pattern = cv2.resize(
                activation_pattern,
                dsize=(img_shape[1], img_shape[0]),
                interpolation=cv2.INTER_CUBIC,
            )
            max_activation_pattern = np.maximum(
                upsampled_activation_pattern, max_activation_pattern
            )

        # show the most highly activated patch of the image by these prototypes
        high_act_patch_indices = find_high_activation_crop(
            max_activation_pattern
        )
        high_act_patch = original_img[
            high_act_patch_indices[0] : high_act_patch_indices[1],
            high_act_patch_indices[2] : high_act_patch_indices[3],
            :,
        ]
        # log("most highly activated patch of the chosen image by these prototypes:")
        # # plt.axis('off')
        # save_image(
        #     os.path.join(
        #         save_analysis_path,
        #         "most_highly_activated_patch_by_%d_prototypes.png" % i,
        #     ),
        #     high_act_patch,
        # )
        # log(
        # "most highly activated patch by these prototypes shown in the original image:"
        # )
        # imsave_with_bbox(
        #     fname=os.path.join(
        #     save_analysis_path,
        #     "most_highly_activated_patch_in_original_img_by_%d_prototypes.png" % i,
        #     ),
        #     img_rgb=original_img,
        #     bbox_height_start=high_act_patch_indices[0],
        #     bbox_height_end=high_act_patch_indices[1],
        #     bbox_width_start=high_act_patch_indices[2],
        #     bbox_width_end=high_act_patch_indices[3],
        #     color=(0, 255, 255),
        # )

        # show the image overlayed with prototypes activation map
        rescaled_activation_pattern = max_activation_pattern - np.amin(
            max_activation_pattern
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
        log("prototypes activation map of the chosen image:")
        # plt.axis('off')
        plt.imsave(
            os.path.join(
                save_analysis_path,
                "prototype_activation_map_by_%d_prototypes.png" % i,
            ),
            overlayed_img,
        )
        log("--------------------------------------------------------------")

    if predicted_cls == correct_cls:
        log("Prediction is correct.")
    else:
        log("Prediction is wrong.")

logclose()
