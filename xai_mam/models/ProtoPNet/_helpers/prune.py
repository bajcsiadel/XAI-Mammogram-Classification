import shutil
from collections import Counter

import numpy as np
import torch

from xai_mam.models.ProtoPNet._helpers.find_nearest import (
    find_k_nearest_patches_to_prototypes,
)


def prune_prototypes(
    dataloader,
    prototype_network_parallel,
    k,
    prune_threshold,
    preprocess_input_function,
    original_model_dir,
    epoch_number,
    logger,
    model_name=None,
    copy_prototype_imgs=True,
):
    # run global analysis
    nearest_train_patch_class_ids = find_k_nearest_patches_to_prototypes(
        dataloader=dataloader,
        prototype_network_parallel=prototype_network_parallel,
        logger=logger,
        k=k,
        preprocess_input_function=preprocess_input_function,
        full_save=False,
    )

    # find prototypes to prune
    original_num_prototypes = prototype_network_parallel.module.__n_prototypes

    prototypes_to_prune = []
    for j in range(prototype_network_parallel.module.__n_prototypes):
        class_j = torch.argmax(
            prototype_network_parallel.module.prototype_class_identity[j]
        ).item()
        nearest_train_patch_class_counts_j = Counter(nearest_train_patch_class_ids[j])
        # if no such element is in Counter, it will return 0
        if nearest_train_patch_class_counts_j[class_j] < prune_threshold:
            prototypes_to_prune.append(j)

    logger.info(f"{k = }, {prune_threshold = }")
    logger.info(f"{len(prototypes_to_prune)} prototypes will be pruned")

    # bookkeeping of prototypes to be pruned
    class_of_prototypes_to_prune = (
        torch.argmax(
            prototype_network_parallel.module.prototype_class_identity[
                prototypes_to_prune
            ],
            dim=1,
        )
        .numpy()
        .reshape(-1, 1)
    )
    prototypes_to_prune_np = np.array(prototypes_to_prune).reshape(-1, 1)
    prune_info = np.hstack((prototypes_to_prune_np, class_of_prototypes_to_prune))

    pruned_prototypes_location = (
        original_model_dir / f"pruned-prototypes-k_{k}-pt_{prune_threshold}"
    )
    pruned_prototypes_location.mkdir(parent=True, exist_ok=True)
    np.save(
        pruned_prototypes_location / "prune_info.npy",
        prune_info,
    )

    # prune prototypes
    prototype_network_parallel.module.prune_prototypes(prototypes_to_prune)
    torch.save(
        obj=prototype_network_parallel.module,
        f=original_model_dir
        / f"pruned_prototypes_epoch{epoch_number}_k{k}_pt{prune_threshold}"
        / model_name
        + "-pruned.pth",
    )
    if copy_prototype_imgs:
        original_img_dir = original_model_dir / "img" / f"epoch-{epoch_number}"
        dst_img_dir = pruned_prototypes_location / "img" / f"epoch-{epoch_number}"
        dst_img_dir.mkdir(parent=True, exist_ok=True)
        prototypes_to_keep = list(
            set(range(original_num_prototypes)) - set(prototypes_to_prune)
        )

        for idx in range(len(prototypes_to_keep)):
            shutil.copyfile(
                src=original_img_dir / f"prototype-img-{prototypes_to_keep[idx]}.png",
                dst=dst_img_dir / f"prototype-img-{idx}.png",
            )

            shutil.copyfile(
                src=original_img_dir
                / f"prototype-img-original-{prototypes_to_keep[idx]}.png",
                dst=dst_img_dir / f"prototype-img-original-{idx}.png",
            )

            shutil.copyfile(
                src=original_img_dir
                / f"prototype-img-original_with_self_act-{prototypes_to_keep[idx]}.png",
                dst=dst_img_dir / f"prototype-img-original_with_self_act-{idx}.png",
            )

            shutil.copyfile(
                src=original_img_dir
                / f"prototype-self-act-{prototypes_to_keep[idx]}.npy",
                dst=dst_img_dir / f"prototype-self-act-{idx}.npy",
            )

            bb = np.load(original_img_dir / f"bb-{epoch_number}.npy")
            bb = bb[prototypes_to_keep]
            np.save(dst_img_dir / f"bb-{epoch_number}.npy", bb)

            bb_rf = np.load(original_img_dir / f"bb-receptive_field-{epoch_number}.npy")
            bb_rf = bb_rf[prototypes_to_keep]
            np.save(
                dst_img_dir / f"bb-receptive_field-{epoch_number}.npy",
                bb_rf,
            )

    return prune_info
