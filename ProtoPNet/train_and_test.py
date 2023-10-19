import time

import numpy as np
import torch
from ProtoPNet.config.settings import (
    prototype_shape,
    separation_type,
    num_classes,
)
from ProtoPNet.util.helpers import list_of_distances
from sklearn.metrics import f1_score


def _train_or_test(
    model,
    dataloader,
    optimizer=None,
    class_specific=True,
    use_l1_mask=True,
    coefs=None,
    use_bce=False,
    log=print,
    backbone_only=False,
):
    """
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    """
    is_train = optimizer is not None
    start = time.time()
    n_examples = 0
    n_correct = 0
    n_batches = 0
    total_cross_entropy = 0
    total_cluster_cost = 0
    # separation cost is meaningful only for class_specific
    total_separation_cost = 0

    true_labels = np.array([])
    predicted_labels = np.array([])

    for image, label in dataloader:
        input = image.cuda()
        target = label.cuda()
        true_labels = np.append(true_labels, label.numpy())

        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            # nn.Module has implemented __call__() function
            # so no need to call .forward
            if backbone_only:
                output = model(input)
            else:
                output, additional_out = model(input)
                min_distances = additional_out["min_distances"]

            # compute loss
            if use_bce:
                one_hot_target = torch.nn.functional.one_hot(
                    target, num_classes
                )
                cross_entropy = (
                    torch.nn.functional.binary_cross_entropy_with_logits(
                        output, one_hot_target.float(), reduction="sum"
                    )
                )
            else:
                cross_entropy = torch.nn.functional.cross_entropy(
                    output, target
                )

            if not backbone_only:
                if class_specific:
                    max_dist = (
                        model.module.prototype_shape[1]
                        * model.module.prototype_shape[2]
                        * model.module.prototype_shape[3]
                    )

                    # prototypes_of_correct_class is a tensor
                    # of shape batch_size * num_prototypes
                    # calculate cluster cost
                    prototypes_of_correct_class = torch.t(
                        model.module.prototype_class_identity[:, label]
                    ).cuda()
                    inverted_distances, target_proto_index = torch.max(
                        (max_dist - min_distances)
                        * prototypes_of_correct_class,
                        dim=1,
                    )
                    cluster_cost = torch.mean(max_dist - inverted_distances)

                    # calculate separation cost
                    prototypes_of_wrong_class = 1 - prototypes_of_correct_class

                    if separation_type == "max":
                        (
                            inverted_distances_to_nontarget_prototypes,
                            _,
                        ) = torch.max(
                            (max_dist - min_distances)
                            * prototypes_of_wrong_class,
                            dim=1,
                        )
                        separation_cost = torch.mean(
                            max_dist
                            - inverted_distances_to_nontarget_prototypes
                        )
                    elif separation_type == "avg":
                        min_distances_detached_prototype_vectors = (
                            model.module.prototype_min_distances(
                                input, detach_prototypes=True
                            )[0]
                        )
                        # calculate avg cluster cost
                        avg_separation_cost = torch.sum(
                            min_distances_detached_prototype_vectors
                            * prototypes_of_wrong_class,
                            dim=1,
                        ) / torch.sum(prototypes_of_wrong_class, dim=1)
                        avg_separation_cost = torch.mean(avg_separation_cost)

                        l2 = (
                            torch.mm(
                                model.module.prototype_vectors[:, :, 0, 0],
                                model.module.prototype_vectors[:, :, 0, 0].t(),
                            )
                            - torch.eye(prototype_shape[0]).cuda()
                        ).norm(p=2)

                        separation_cost = avg_separation_cost
                    elif separation_type == "margin":
                        # For each input get the distance
                        # to the closest target class prototype
                        min_distance_target = (
                            max_dist - inverted_distances.reshape((-1, 1))
                        )

                        all_distances = additional_out["distances"]
                        min_indices = additional_out["min_indices"]

                        anchor_index = min_indices[
                            torch.arange(
                                0, target_proto_index.size(0), dtype=torch.long
                            ),
                            target_proto_index,
                        ].squeeze()
                        all_distances = all_distances.view(
                            all_distances.size(0), all_distances.size(1), -1
                        )
                        distance_at_anchor = all_distances[
                            torch.arange(
                                0, all_distances.size(0), dtype=torch.long
                            ),
                            :,
                            anchor_index,
                        ]

                        # For each non-target prototype
                        # compute difference compared to closest target prototype
                        # d(a, p) - d(a, n) term from TripletMarginLoss
                        distance_pos_neg = (
                            min_distance_target - distance_at_anchor
                        ) * prototypes_of_wrong_class
                        # Separation cost is the margin loss
                        # max(d(a, p) - d(a, n) + margin, 0)
                        separation_cost = torch.mean(
                            torch.maximum(
                                distance_pos_neg + coefs["sep_margin"],
                                torch.tensor(
                                    0.0, device=distance_pos_neg.device
                                ),
                            )
                        )
                    else:
                        raise ValueError(
                            f"""
                                separation_type has to be one of [max, mean, margin],
                                got {separation_type}
                            """
                        )

                    # print(separation_cost.item())

                    if use_l1_mask:
                        l1_mask = (
                            1
                            - torch.t(
                                model.module.prototype_class_identity
                            ).cuda()
                        )
                        l1 = (model.module.last_layer.weight * l1_mask).norm(
                            p=1
                        )
                    else:
                        l1 = model.module.last_layer.weight.norm(p=1)

                else:
                    min_distance, _ = torch.min(min_distances, dim=1)
                    cluster_cost = torch.mean(min_distance)
                    l1 = model.module.last_layer.weight.norm(p=1)
            else:
                l1 = model.module.last_layer.weight.norm(p=1)

            # evaluation statistics
            _, predicted = torch.max(output.data, 1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()

            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            if not backbone_only:
                total_cluster_cost += cluster_cost.item()
                total_separation_cost += separation_cost.item()

        # compute gradient and do SGD step
        if is_train:
            if backbone_only:
                if coefs is not None:
                    loss = coefs["crs_ent"] * cross_entropy + coefs["l1"] * l1
                else:
                    loss = cross_entropy + 1e-4 * l1
            else:
                if class_specific:
                    if coefs is not None:
                        loss = (
                            coefs["crs_ent"] * cross_entropy
                            + coefs["clst"] * cluster_cost
                            + coefs["sep"] * separation_cost
                            + (
                                coefs["l2"] * l2
                                if separation_type == "avg"
                                else 0
                            )
                            + coefs["l1"] * l1
                        )
                    else:
                        loss = (
                            cross_entropy
                            + 0.8 * cluster_cost
                            - 0.08 * separation_cost
                            + 1e-4 * l1
                        )
                else:
                    if coefs is not None:
                        loss = (
                            coefs["crs_ent"] * cross_entropy
                            + coefs["clst"] * cluster_cost
                            + coefs["l1"] * l1
                        )
                    else:
                        loss = cross_entropy + 0.8 * cluster_cost + 1e-4 * l1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        predicted_labels = np.append(predicted_labels, predicted.cpu().numpy())

        del input
        del target
        del output
        del predicted
        if not backbone_only:
            del min_distances

    end = time.time()

    log("\ttime: \t{0}".format(end - start))
    log("\tcross ent: \t{0}".format(total_cross_entropy / n_batches))
    if not backbone_only:
        log("\tcluster: \t{0}".format(total_cluster_cost / n_batches))
        if class_specific:
            log("\tseparation:\t{0}".format(total_separation_cost / n_batches))
    log("\taccu: \t\t{0}%".format(n_correct / n_examples * 100))
    log(
        "\tmicro f1: \t\t{0}".format(
            f1_score(true_labels, predicted_labels, average="micro")
        )
    )
    log(
        "\tmacro f1: \t\t{0}".format(
            f1_score(true_labels, predicted_labels, average="macro")
        )
    )
    log(
        "\tl1: \t\t{0}".format(model.module.last_layer.weight.norm(p=1).item())
    )
    if not backbone_only:
        p = model.module.prototype_vectors.view(
            model.module.num_prototypes, -1
        ).cpu()
        with torch.no_grad():
            p_avg_pair_dist = torch.mean(list_of_distances(p, p))
        log("\tp dist pair: \t{0}".format(p_avg_pair_dist.item()))

    return n_correct / n_examples


def train(
    model,
    dataloader,
    optimizer,
    class_specific=False,
    coefs=None,
    use_bce=False,
    log=print,
    backbone_only=False,
):
    assert optimizer is not None

    log("\ttrain")
    model.train()
    return _train_or_test(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        class_specific=class_specific,
        coefs=coefs,
        use_bce=use_bce,
        log=log,
        backbone_only=backbone_only,
    )


def test(
    model,
    dataloader,
    class_specific=False,
    coefs=None,
    use_bce=False,
    log=print,
    backbone_only=False,
):
    log("\ttest")
    model.eval()
    return _train_or_test(
        model=model,
        dataloader=dataloader,
        optimizer=None,
        class_specific=class_specific,
        coefs=coefs,
        use_bce=use_bce,
        log=log,
        backbone_only=backbone_only,
    )


def last_only(model, log=print, backbone_only=False):
    for p in model.module.features.parameters():
        p.requires_grad = False
    if not backbone_only:
        for p in model.module.add_on_layers.parameters():
            p.requires_grad = False
        model.module.prototype_vectors.requires_grad = False
    for p in model.module.last_layer.parameters():
        p.requires_grad = True

    log("\tlast layer")


def warm_only(model, log=print, backbone_only=False):
    for p in model.module.features.parameters():
        p.requires_grad = False
    if not backbone_only:
        for p in model.module.add_on_layers.parameters():
            p.requires_grad = True
        model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True

    log("\twarm")


def joint(model, log=print, backbone_only=False):
    for p in model.module.features.parameters():
        p.requires_grad = True
    if not backbone_only:
        for p in model.module.add_on_layers.parameters():
            p.requires_grad = True
        model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True

    log("\tjoint")
