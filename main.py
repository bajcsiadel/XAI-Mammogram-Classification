import json
import os
import pprint
import re
import shutil
import torch

# import model as model
# import push as push
# import train_and_test as tnt
# import util.save as save
# import torch
# from database.dataloader import (
#     train_batch_size,
#     train_loader,
#     train_push_loader,
#     valid_loader,
# )

# from ProtoPNet import model

from ProtoPNet.util import helpers
from ProtoPNet.util import args
from ProtoPNet.util.log import Log
# from ProtoPNet.util.preprocess import preprocess_input_function


def main(args, logger):
    # set used GPU id
    torch.cuda.set_device(args.gpu_id)
    logger.log_info(f"Visible devices set to: {torch.cuda.current_device()}")
    exit()
    # create result directory
    log_source_dir = os.path.join(
        args.log_dir,
        "src",
    )
    helpers.makedir(log_source_dir)

    json.dump(settings, open(os.path.join(log_source_dir, "settings.json"), "w"), indent=4)

    # copy code to result directory
    shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=log_source_dir)
    base_architecture_type = re.match("^[a-z]*", settings["base_architecture"]).group(0)
    shutil.copy(
        src=os.path.join(
            os.getcwd(),
            "ProtoPNet",
            "config",
            "backbone_features",
            base_architecture_type + "_features.py",
        ),
        dst=log_source_dir,
    )
    shutil.copy(src=os.path.join(os.getcwd(), "ProtoPNet", "model.py"), dst=log_source_dir)
    shutil.copy(
        src=os.path.join(os.getcwd(), "ProtoPNet", "train_and_test.py"), dst=log_source_dir
    )
    model_dir = os.path.join(args.log_dir, "model")
    helpers.makedir(model_dir)
    img_dir = os.path.join(args.log_dir, "img")
    helpers.makedir(img_dir)

    # weight_matrix_filename = "outputL_weights"
    prototype_img_filename_prefix = "prototype-img"
    prototype_self_act_filename_prefix = "prototype-self-act"
    proto_bound_boxes_filename_prefix = "bb"

    # we should look into distributed sampler more carefully
    # at torch.utils.data.distributed.DistributedSampler(train_dataset)
    logger.log_info(f"training set size: {len(train_loader.dataset)}")
    logger.log_info(f"push set size: {len(train_push_loader.dataset)}")
    logger.log_info(f"test set size: {len(valid_loader.dataset)}")
    logger.log_info(f"batch size: {train_batch_size}")
    logger.log_info(f"number of prototypes per class: {settings['num_prototypes_per_class']}")

    # construct the model
    ppnet = model.construct_PPNet(
        base_architecture=settings["base_architecture"],
        pretrained=settings["use_pretrain"],
        img_shape=dataset_config["img_shape"],
        prototype_shape=settings["prototype_shape"],
        num_classes=num_classes,
        prototype_activation_function=prototype_activation_function,
        add_on_layers_type=add_on_layers_type,
        backbone_only=backbone_only,
        positive_weights_in_classifier=args.pos_weights_in_classifier
    )
    # if prototype_activation_function == 'linear':
    #    ppnet.set_last_layer_incorrect_connection(incorrect_strength=0)
    ppnet = ppnet.cuda()
    ppnet_multi = torch.nn.DataParallel(ppnet)
    class_specific = True

    joint_optimizer_specs = [
        {
            "params": ppnet.features.parameters(),
            "lr": joint_optimizer_lrs["features"],
            "weight_decay": 1e-3,
        },  # bias are now also being regularized
        {
            "params": ppnet.add_on_layers.parameters(),
            "lr": joint_optimizer_lrs["add_on_layers"],
            "weight_decay": 1e-3,
        },
    ]
    if not backbone_only:
        joint_optimizer_specs += [
            {
                "params": ppnet.prototype_vectors,
                "lr": joint_optimizer_lrs["prototype_vectors"],
            },
        ]

    joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
    joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(
        joint_optimizer, step_size=joint_lr_step_size, gamma=0.1
    )

    from ProtoPNet.config.settings import warm_optimizer_lrs

    warm_optimizer_specs = [
        {
            "params": ppnet.add_on_layers.parameters(),
            "lr": warm_optimizer_lrs["add_on_layers"],
            "weight_decay": 1e-3,
        },
    ]
    if not backbone_only:
        warm_optimizer_specs += [
            {
                "params": ppnet.prototype_vectors,
                "lr": warm_optimizer_lrs["prototype_vectors"],
            },
        ]
    warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

    from ProtoPNet.config.settings import last_layer_optimizer_lr

    last_layer_optimizer_specs = [
        {
            "params": ppnet.last_layer.parameters(),
            "lr": last_layer_optimizer_lr,
        }
    ]
    last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

    # train the model
    log("start training")

    for epoch in range(num_train_epochs):
        log("epoch: \t{0}".format(epoch))

        if not backbone_only and epoch < num_warm_epochs:
            tnt.warm_only(
                model=ppnet_multi, log=log, backbone_only=backbone_only
            )
            _ = tnt.train(
                model=ppnet_multi,
                dataloader=train_loader,
                optimizer=warm_optimizer,
                class_specific=class_specific,
                coefs=coefs,
                use_bce=use_binary_cross_entropy,
                log=log,
                backbone_only=backbone_only,
            )
        else:
            tnt.joint(model=ppnet_multi, log=log, backbone_only=backbone_only)
            if epoch > 0:
                joint_lr_scheduler.step()
            _ = tnt.train(
                model=ppnet_multi,
                dataloader=train_loader,
                optimizer=joint_optimizer,
                class_specific=class_specific,
                coefs=coefs,
                use_bce=use_binary_cross_entropy,
                log=log,
                backbone_only=backbone_only,
            )

        accu = tnt.test(
            model=ppnet_multi,
            dataloader=valid_loader,
            class_specific=class_specific,
            coefs=coefs,
            use_bce=use_binary_cross_entropy,
            log=log,
            backbone_only=backbone_only,
        )
        save.save_model_w_condition(
            model=ppnet,
            model_dir=model_dir,
            model_name=str(epoch) + "nopush",
            accu=accu,
            target_accu=0.60,
            log=log,
        )

        if not backbone_only and epoch >= push_start and epoch in push_epochs:
            push.push_prototypes(
                train_push_loader,  # pytorch dataloader (must be unnormalized in [0,1])
                prototype_network_parallel=ppnet_multi,
                # pytorch network with prototype_vectors
                class_specific=class_specific,
                preprocess_input_function=preprocess_input_function,  # normalize
                prototype_layer_stride=1,
                root_dir_for_saving_prototypes=img_dir,
                # if not None, prototypes will be saved here
                epoch_number=epoch,
                # if not provided, prototypes saved previously will be overwritten
                prototype_img_filename_prefix=prototype_img_filename_prefix,
                prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
                save_prototype_class_identity=True,
                log=log,
            )
            accu = tnt.test(
                model=ppnet_multi,
                dataloader=valid_loader,
                class_specific=class_specific,
                coefs=coefs,
                use_bce=use_binary_cross_entropy,
                log=log,
            )
            save.save_model_w_condition(
                model=ppnet,
                model_dir=model_dir,
                model_name=str(epoch) + "push",
                accu=accu,
                target_accu=0.60,
                log=log,
            )

            if prototype_activation_function != "linear":
                tnt.last_only(model=ppnet_multi, log=log)
                for i in range(num_last_layer_train_epochs):
                    log("iteration: \t{0}".format(i))
                    _ = tnt.train(
                        model=ppnet_multi,
                        dataloader=train_loader,
                        optimizer=last_layer_optimizer,
                        class_specific=class_specific,
                        coefs=coefs,
                        use_bce=use_binary_cross_entropy,
                        log=log,
                    )
                    accu = tnt.test(
                        model=ppnet_multi,
                        dataloader=valid_loader,
                        class_specific=class_specific,
                        coefs=coefs,
                        use_bce=use_binary_cross_entropy,
                        log=log,
                    )
                    save.save_model_w_condition(
                        model=ppnet,
                        model_dir=model_dir,
                        model_name=str(epoch) + "_" + str(i) + "push",
                        accu=accu,
                        target_accu=0.60,
                        log=log,
                    )

    logclose()


if __name__ == "__main__":
    command_line_params = args.get_args()
    logger = Log(command_line_params.log_dir)
    args.save_args(command_line_params, logger.metadata_dir)

    try:
        main(command_line_params, logger)
    except Exception as e:
        logger.log_exception(e)
