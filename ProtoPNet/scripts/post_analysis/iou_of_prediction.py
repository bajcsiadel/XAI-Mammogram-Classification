import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.datasets as datasets
from database.config import pascalvoc_config, pascalvoc_dir, test_batch_size
from database.dataloader import base_transform, test_loader
from ProtoPNet.util.helpers import iou, load_model_parallel, makedir

nppc = pascalvoc_config["num_prototypes_per_class"]


def get_test_mask_dl():
    test_mask_dir = os.path.join(pascalvoc_dir, "test_mask")

    test_mask_set = datasets.ImageFolder(
        test_mask_dir,
        base_transform,
    )

    return torch.utils.data.DataLoader(
        test_mask_set,
        batch_size=test_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=False,
    )


def calculate_iou_scores(
    test_loader, test_mask_loader, ppnet_multi, treshold=[0.0]
):
    iou_scores = np.empty((len(test_loader.dataset), len(treshold)))
    i = 0

    for (test_imgs, test_labels), (img_masks, _) in zip(
        test_loader, test_mask_loader
    ):
        with torch.no_grad():
            test_imgs = test_imgs.cuda()
            (
                min_distances,
                distances,
                min_indices,
            ) = ppnet_multi.module.prototype_min_distances(test_imgs)
        min_indices = torch.squeeze(min_indices, dim=3)
        min_indices = torch.squeeze(min_indices, dim=2)
        max_similarities = ppnet_multi.module.distance_2_similarity(
            min_distances
        )
        map_shape = distances.shape[2:]
        img_masks = img_masks[:, 0, :, :].squeeze()

        # iterate on batch
        for test_label, max_sim, min_idx, img_mask in zip(
            test_labels, max_similarities, min_indices, img_masks
        ):
            correct_class_slice = slice(
                test_label * nppc, (test_label + 1) * nppc
            )
            correct_class_prototype_weights = (
                ppnet_multi.module.last_layer.weight[
                    test_label, correct_class_slice
                ]
            )

            sim_map = torch.zeros(map_shape).cuda().flatten()
            for idx, sim, weight in zip(
                min_idx[correct_class_slice],
                max_sim[correct_class_slice],
                correct_class_prototype_weights,
            ):
                sim_map[idx] += sim * weight

            sim_map = sim_map.reshape(map_shape)
            upsampled_sim_map = cv2.resize(
                sim_map.cpu().detach().numpy(),
                dsize=test_imgs.shape[2:],
                interpolation=cv2.INTER_CUBIC,
            )
            upsampled_sim_map = upsampled_sim_map - np.amin(upsampled_sim_map)
            upsampled_sim_map = upsampled_sim_map / np.amax(upsampled_sim_map)

            for j, th in enumerate(treshold):
                iou_scores[i][j] = iou(
                    img_mask.cpu().detach().numpy(), upsampled_sim_map > th
                )
            i += 1
    return iou_scores


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid[0]
    load_model_dir = args.modeldir[0]
    load_model_name = args.model[0]
    treshold = np.array(args.treshold)

    ppnet_multi = load_model_parallel(load_model_dir, load_model_name)
    test_mask_loader = get_test_mask_dl()

    iou_scores = calculate_iou_scores(
        test_loader, test_mask_loader, ppnet_multi, treshold
    )

    savefig_dir = os.path.join(load_model_dir, "iou")
    makedir(savefig_dir)
    savefig_path = os.path.join(
        savefig_dir, load_model_name.replace(".pth", ".png")
    )

    plt.boxplot(iou_scores)
    plt.xlabel("treshold")
    plt.xticks(np.arange(len(treshold)) + 1, treshold)
    plt.ylabel("iou score")
    plt.savefig(savefig_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-gpuid", nargs=1, type=str, default="0")
    parser.add_argument("-modeldir", nargs=1, type=str)
    parser.add_argument("-model", nargs=1, type=str)
    parser.add_argument("-treshold", nargs="*", type=float, default=0.0)
    args = parser.parse_args()

    main(args)
