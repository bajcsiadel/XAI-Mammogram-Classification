import time

import numpy as np
import torch
from sklearn.metrics import f1_score

from ProtoPNet.util.helpers import list_of_distances


def _train_or_test(
    model,
    dataloader,
    prototype_shape,
    separation_type,
    number_of_classes,
    step,
    log,
    optimizer=None,
    class_specific=True,
    use_l1_mask=True,
    loss_coefficients=None,
    use_bce=False,
    backbone_only=False,
    device="cpu",
):
    """

    :param model: the multi-gpu model
    :param dataloader:
    :param prototype_shape:
    :type prototype_shape: tuple
    :param separation_type:
    :type separation_type: str
    :param number_of_classes:
    :param optimizer: optimizer used during training. If ``None``, then it
        is a test evaluation
    :param class_specific: Defaults to ``True``
    :type class_specific: bool
    :param use_l1_mask: Defaults to ``True``
    :type use_l1_mask: bool
    :param loss_coefficients:
    :type loss_coefficients: dict
    :param use_bce: Defaults to ``False``
    :type use_bce: bool
    :param log:
    :type log: ProtoPNet.util.log.Log
    :param backbone_only: Defaults to ``False``
    :type backbone_only: bool
    :return:
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

    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    for image, label in dataloader:
        input_ = image.to(device)
        target_ = label.to(device)
        true_labels = np.append(true_labels, label.numpy())

        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            # nn.Module has implemented __call__() function
            # so no need to call .forward
            if backbone_only:
                output = model(input_)
            else:
                output, additional_out = model(input_)
                min_distances = additional_out["min_distances"]

            # compute loss
            if use_bce:
                one_hot_target = torch.nn.functional.one_hot(target_, number_of_classes)
                cross_entropy = torch.nn.functional.binary_cross_entropy_with_logits(
                    output, one_hot_target.float(), reduction="sum"
                )
            else:
                cross_entropy = torch.nn.functional.cross_entropy(output, target_)

            if not backbone_only:
                if class_specific:
                    max_dist = (
                        model.prototype_shape[1]
                        * model.prototype_shape[2]
                        * model.prototype_shape[3]
                    )

                    # prototypes_of_correct_class is a tensor
                    # of shape batch_size * num_prototypes
                    # calculate cluster cost
                    prototypes_of_correct_class = torch.t(
                        model.prototype_class_identity[:, label]
                    ).to(device)
                    inverted_distances, target_proto_index = torch.max(
                        (max_dist - min_distances) * prototypes_of_correct_class,
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
                            (max_dist - min_distances) * prototypes_of_wrong_class,
                            dim=1,
                        )
                        separation_cost = torch.mean(
                            max_dist - inverted_distances_to_nontarget_prototypes
                        )
                    elif separation_type == "avg":
                        min_distances_detached_prototype_vectors = (
                            model.prototype_min_distances(
                                input_, detach_prototypes=True
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
                                model.prototype_vectors[:, :, 0, 0],
                                model.prototype_vectors[:, :, 0, 0].t(),
                            )
                            - torch.eye(prototype_shape[0]).to(device)
                        ).norm(p=2)

                        separation_cost = avg_separation_cost
                    elif separation_type == "margin":
                        # For each input get the distance
                        # to the closest target class prototype
                        min_distance_target = max_dist - inverted_distances.reshape(
                            (-1, 1)
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
                            torch.arange(0, all_distances.size(0), dtype=torch.long),
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
                                distance_pos_neg
                                + loss_coefficients["separation_margin"],
                                torch.tensor(0.0, device=distance_pos_neg.device),
                            )
                        )
                    else:
                        raise ValueError(
                            f"""
                                separation_type has to be one of [max, mean, margin],
                                got {separation_type}
                            """
                        )

                    if use_l1_mask:
                        l1_mask = (
                            1 - torch.t(model.prototype_class_identity).to(device)
                        )
                        l1 = (model.last_layer.weight * l1_mask).norm(p=1)
                    else:
                        l1 = model.last_layer.weight.norm(p=1)

                else:
                    min_distance, _ = torch.min(min_distances, dim=1)
                    cluster_cost = torch.mean(min_distance)
                    l1 = model.last_layer.weight.norm(p=1)
            else:
                l1 = model.last_layer.weight.norm(p=1)

            # evaluation statistics
            _, predicted = torch.max(output.data, 1)
            n_examples += target_.size(0)
            n_correct += (predicted == target_).sum().item()

            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            if not backbone_only:
                total_cluster_cost += cluster_cost.item()
                total_separation_cost += separation_cost.item()

        # compute gradient and do SGD step
        if is_train:
            if backbone_only:
                if loss_coefficients is not None:
                    loss = (
                        loss_coefficients["cross_entropy"] * cross_entropy
                        + loss_coefficients["l1"] * l1
                    )
                else:
                    loss = cross_entropy + 1e-4 * l1
            else:
                if class_specific:
                    if loss_coefficients is not None:
                        loss = (
                            loss_coefficients["cross_entropy"] * cross_entropy
                            + loss_coefficients["clustering"] * cluster_cost
                            + loss_coefficients["separation"] * separation_cost
                            + (
                                loss_coefficients["l2"] * l2
                                if separation_type == "avg"
                                else 0
                            )
                            + loss_coefficients["l1"] * l1
                        )
                    else:
                        loss = (
                            cross_entropy
                            + 0.8 * cluster_cost
                            - 0.08 * separation_cost
                            + 1e-4 * l1
                        )
                else:
                    if loss_coefficients is not None:
                        loss = (
                            loss_coefficients["cross_entropy"] * cross_entropy
                            + loss_coefficients["clustering"] * cluster_cost
                            + loss_coefficients["l1"] * l1
                        )
                    else:
                        loss = cross_entropy + 0.8 * cluster_cost + 1e-4 * l1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        predicted_labels = np.append(predicted_labels, predicted.cpu().numpy())

        del input_
        del target_
        del output
        del predicted
        if not backbone_only:
            del min_distances

    end = time.time()

    total_time = end - start
    cross_entropy = total_cross_entropy / n_batches
    cluster_cost = total_cluster_cost / n_batches if not backbone_only else None
    separation_cost = (
        total_separation_cost / n_batches
        if not backbone_only and class_specific
        else None
    )
    accuracy = n_correct / n_examples
    micro_f1 = f1_score(true_labels, predicted_labels, average="micro")
    macro_f1 = f1_score(true_labels, predicted_labels, average="macro")
    l1_norm = model.last_layer.weight.norm(p=1).item()

    p_avg_pair_dist = None
    if not backbone_only:
        p = model.prototype_vectors.view(model.num_prototypes, -1).cpu()
        with torch.no_grad():
            p_avg_pair_dist = torch.mean(list_of_distances(p, p)).item()

    log(f"INFO: \t\t\t\t{'time: ':<13}{total_time}")
    log(f"INFO: \t\t\t\t{'cross ent: ':<13}{cross_entropy}")
    if not backbone_only:
        log(f"INFO: \t\t\t\t{'cluster: ':<13}{cluster_cost}")
        if class_specific:
            log(f"INFO: \t\t\t\t{'separation: ':<13}{separation_cost}")
    log(f"INFO: \t\t\t\t{'accu: ':<13}{accuracy:.2%}")
    log(f"INFO: \t\t\t\t{'micro f1: ':<13}{micro_f1:.2%}")
    log(f"INFO: \t\t\t\t{'macro f1: ':<13}{macro_f1:.2%}")
    log(f"INFO: \t\t\t\t{'l1: ':<13}{l1_norm}")
    if not backbone_only:
        log(f"INFO: \t\t\t\t{'p dist pair: ':<13}{p_avg_pair_dist}")

    if hasattr(log, "csv_log_values"):
        log.csv_log_values(
            "train_model",
            total_time,
            cross_entropy,
            cluster_cost,
            separation_cost,
            accuracy,
            micro_f1,
            macro_f1,
            l1_norm,
            p_avg_pair_dist,
        )

        if is_train:
            writer = log.tensorboard.train
        else:
            writer = log.tensorboard.validation

        writer.add_scalar("accuracy", accuracy, step)
        write_loss = {
            "cross_entropy": cross_entropy,
            "l1": l1_norm,
        }

        if is_train:
            write_loss["loss"] = loss.item()

        if not backbone_only:
            write_loss["cluster_cost"] = cluster_cost
            if class_specific:
                write_loss["separation_cost"] = separation_cost
        writer.add_scalars("loss", write_loss, step)

    return accuracy


def test(
    model,
    dataloader,
    prototype_shape,
    separation_type,
    number_of_classes,
    step,
    log,
    class_specific=False,
    loss_coefficients=None,
    use_bce=False,
    backbone_only=False,
    device="cpu",
):
    log("INFO: \t\t\ttest")
    model.eval()
    return _train_or_test(
        model=model,
        dataloader=dataloader,
        prototype_shape=prototype_shape,
        separation_type=separation_type,
        number_of_classes=number_of_classes,
        optimizer=None,
        step=step,
        log=log,
        class_specific=class_specific,
        loss_coefficients=loss_coefficients,
        use_bce=use_bce,
        backbone_only=backbone_only,
        device=device,
    )


def train(
    model,
    dataloader,
    prototype_shape,
    separation_type,
    number_of_classes,
    optimizer,
    step,
    log,
    class_specific=False,
    loss_coefficients=None,
    use_bce=False,
    backbone_only=False,
    device="cpu",
):
    assert optimizer is not None

    log("INFO: \t\t\ttrain")
    model.train()
    return _train_or_test(
        model=model,
        dataloader=dataloader,
        prototype_shape=prototype_shape,
        separation_type=separation_type,
        number_of_classes=number_of_classes,
        optimizer=optimizer,
        step=step,
        log=log,
        class_specific=class_specific,
        loss_coefficients=loss_coefficients,
        use_bce=use_bce,
        backbone_only=backbone_only,
        device=device,
    )


def last_only(model, log=print, backbone_only=False):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    for p in model.features.parameters():
        p.requires_grad = False
    if not backbone_only:
        for p in model.add_on_layers.parameters():
            p.requires_grad = False
        model.prototype_vectors.requires_grad = False
    for p in model.last_layer.parameters():
        p.requires_grad = True

    log("INFO: \tlast layer")


def warm_only(model, log=print, backbone_only=False):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    for p in model.module.features.parameters():
        p.requires_grad = False
    if not backbone_only:
        for p in model.add_on_layers.parameters():
            p.requires_grad = True
        model.prototype_vectors.requires_grad = True
    for p in model.last_layer.parameters():
        p.requires_grad = True

    log("INFO: \twarm")


def joint(model, log=print, backbone_only=False):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    for p in model.features.parameters():
        p.requires_grad = True
    if not backbone_only:
        for p in model.add_on_layers.parameters():
            p.requires_grad = True
        model.prototype_vectors.requires_grad = True
    for p in model.last_layer.parameters():
        p.requires_grad = True

    log("INFO: \tjoint")
