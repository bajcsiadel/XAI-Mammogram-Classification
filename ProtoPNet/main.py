import gin
import numpy as np
import os
import re
import shutil
import torch

from dotenv import load_dotenv

load_dotenv()

from ProtoPNet import model
from ProtoPNet import push
from ProtoPNet import train_and_test as tnt

from ProtoPNet.dataset.metadata import DATASETS

from ProtoPNet.util import args
from ProtoPNet.util import helpers
from ProtoPNet.util.log import Log
from ProtoPNet.util import save

from ProtoPNet.util.preprocess import preprocess


@gin.configurable
def main(args, logger, dataset_module):
    # set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # set used GPU id
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    logger.log_info(f"Visible devices set to: {torch.cuda.current_device()}")

    # create result directory
    log_source_dir = os.path.join(
        args.log_dir,
        "src",
    )
    helpers.makedir(log_source_dir)

    dataset_information = DATASETS[args.dataset]

    shutil.copy(dataset_information.METADATA.FILE, logger.metadata_dir)

    # copy code to result directory
    base_architecture_type = re.match("^[a-z]*", args.backbone).group(0)
    shutil.copy(
        src=os.path.join(
            os.getenv("PROJECT_ROOT"),
            "ProtoPNet",
            "config",
            "backbone_features",
            base_architecture_type + "_features.py",
        ),
        dst=log_source_dir,
    )
    shutil.copy(
        src=os.path.join(
            os.getenv("PROJECT_ROOT"),
            "ProtoPNet",
            "model.py"
        ),
        dst=log_source_dir
    )
    shutil.copy(
        src=os.path.join(
            os.getenv("PROJECT_ROOT"),
            "ProtoPNet",
            "train_and_test.py"
        ),
        dst=log_source_dir
    )

    model_dir = os.path.join(args.log_dir, "model")
    helpers.makedir(model_dir)
    img_dir = os.path.join(args.log_dir, "img")
    helpers.makedir(img_dir)

    # weight_matrix_filename = "outputL_weights"
    prototype_img_filename_prefix = "prototype-img"
    prototype_self_act_filename_prefix = "prototype-self-act"
    proto_bound_boxes_filename_prefix = "bb"

    if args.cross_validation_folds > 1:
        logger.log_info("using cross-validation with:\n"
                        f"\t{args.cross_validation_folds} folds")
        if args.stratified_cross_validation:
            logger.log_info(f"\tstratified")
        if args.grouped_cross_validation:
            logger.log_info(f"\tgrouped")

    logger.log_info(f"number of prototypes per class: {args.prototypes_per_class}")

    # train the model
    logger.log_info("start training")

    for fold, (train_sampler, validation_sampler) in dataset_module.folds:
        # construct the model
        ppnet = model.construct_PPNet()
        ppnet = ppnet.cuda()
        ppnet_multi = torch.nn.DataParallel(ppnet)
        class_specific = True

        joint_optimizer_specs = [
            {
                "params": ppnet.features.parameters(),
                "lr": args.joint_lr_features,
                "weight_decay": 1e-3,
            },  # bias are now also being regularized
            {
                "params": ppnet.add_on_layers.parameters(),
                "lr": args.joint_lr_add_on_layers,
                "weight_decay": 1e-3,
            },
        ]
        if not args.backbone_only:
            joint_optimizer_specs += [
                {
                    "params": ppnet.prototype_vectors,
                    "lr": args.joint_lr_prototype_vectors,
                },
            ]

        joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
        joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            joint_optimizer, step_size=args.joint_lr_step_size, gamma=0.1
        )

        warm_optimizer_specs = [
            {
                "params": ppnet.add_on_layers.parameters(),
                "lr": args.warm_lr_add_on_layers,
                "weight_decay": 1e-3,
            },
        ]
        if not args.backbone_only:
            warm_optimizer_specs += [
                {
                    "params": ppnet.prototype_vectors,
                    "lr": args.warm_lr_prototype_vectors,
                },
            ]
        warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

        last_layer_optimizer_specs = [
            {
                "params": ppnet.last_layer.parameters(),
                "lr": args.last_layer_lr,
            }
        ]
        last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

        logger.log_info(f"\tFOLD {fold}")
        logger.log_info(f"\t\ttrain set size: {len(train_sampler)}")
        logger.log_info(f"\t\tvalidation set size: {len(validation_sampler)}")

        if not args.backbone_only:
            logger.log_info(f"\t\tbatch size: {args.batch_size_pretrain}")
            tnt.warm_only(model=ppnet_multi, log=logger, backbone_only=args.backbone_only)

            train_loader = dataset_module.train_dataloader(
                sampler=train_sampler,
                batch_size=args.batch_size_pretrain,
            )
            validation_loader = dataset_module.train_dataloader(
                sampler=validation_sampler,
                batch_size=args.batch_size_pretrain,
            )

            for epoch in range(args.epochs_pretrain):
                logger.log_info(f"\t\twarm epoch: \t{epoch + 1}")
                logger.csv_log_index("train_model", (fold, epoch + 1, "warm train"))
                _ = tnt.train(
                    model=ppnet_multi,
                    dataloader=train_loader,
                    prototype_shape=args.prototype_shape,
                    separation_type=args.separation_type,
                    number_of_classes=args.number_of_classes,
                    optimizer=warm_optimizer,
                    class_specific=class_specific,
                    loss_coefficients=args.loss_coefficients,
                    use_bce=args.binary_cross_entropy,
                    log=logger,
                    backbone_only=args.backbone_only,
                )

                logger.csv_log_index("train_model", (fold, epoch + 1, "warm validation"))
                accu = tnt.test(
                    model=ppnet_multi,
                    dataloader=validation_loader,
                    prototype_shape=args.prototype_shape,
                    separation_type=args.separation_type,
                    number_of_classes=args.number_of_classes,
                    class_specific=class_specific,
                    loss_coefficients=args.loss_coefficients,
                    use_bce=args.binary_cross_entropy,
                    log=logger,
                    backbone_only=args.backbone_only,
                )
                save.save_model_w_condition(
                    model=ppnet,
                    model_dir=model_dir,
                    model_name=f"{epoch}-warm",
                    accu=accu,
                    target_accu=0.60,
                    log=logger,
                )

        tnt.joint(model=ppnet_multi, log=logger, backbone_only=args.backbone_only)

        train_loader = dataset_module.train_dataloader(
            sampler=train_sampler,
            batch_size=args.batch_size,
        )
        validation_loader = dataset_module.train_dataloader(
            sampler=validation_sampler,
            batch_size=args.batch_size,
        )
        push_loader = dataset_module.push_dataloader(
            sampler=train_sampler,
            batch_size=args.batch_size_push,
        )

        for epoch in range(args.epochs):
            logger.log_info(f"\t\twarm epoch: \t{epoch + 1 + args.epochs_pretrain}")
            if epoch > 0:
                joint_lr_scheduler.step()

            logger.csv_log_index("train_model", (fold, epoch + 1 + args.epochs_pretrain, "train"))
            _ = tnt.train(
                model=ppnet_multi,
                dataloader=train_loader,
                prototype_shape=args.prototype_shape,
                separation_type=args.separation_type,
                number_of_classes=args.number_of_classes,
                optimizer=joint_optimizer,
                class_specific=class_specific,
                loss_coefficients=args.loss_coefficients,
                use_bce=args.binary_cross_entropy,
                log=logger,
                backbone_only=args.backbone_only,
            )

            logger.csv_log_index("train_model", (fold, epoch + 1 + args.epochs_pretrain, "validation"))
            accu = tnt.test(
                model=ppnet_multi,
                dataloader=validation_loader,
                prototype_shape=args.prototype_shape,
                separation_type=args.separation_type,
                number_of_classes=args.number_of_classes,
                class_specific=class_specific,
                loss_coefficients=args.loss_coefficients,
                use_bce=args.binary_cross_entropy,
                log=logger,
                backbone_only=args.backbone_only,
            )
            save.save_model_w_condition(
                model=ppnet,
                model_dir=model_dir,
                model_name=f"{epoch}-no_push",
                accu=accu,
                target_accu=0.60,
                log=logger,
            )
            if not args.backbone_only and epoch in args.push_epochs:
                push.push_prototypes(
                    push_loader,  # pytorch dataloader (must be unnormalized in [0,1])
                    prototype_network_parallel=ppnet_multi,
                    # pytorch network with prototype_vectors
                    class_specific=class_specific,
                    preprocess_input_function=preprocess,  # normalize
                    prototype_layer_stride=1,
                    root_dir_for_saving_prototypes=img_dir,
                    # if not None, prototypes will be saved here
                    epoch_number=epoch,
                    # if not provided, prototypes saved previously will be overwritten
                    prototype_img_filename_prefix=prototype_img_filename_prefix,
                    prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                    proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
                    save_prototype_class_identity=True,
                    log=logger,
                )

                logger.csv_log_index("train_model", (fold, epoch + 1 + args.epochs_pretrain, "push validation"))
                accu = tnt.test(
                    model=ppnet_multi,
                    dataloader=validation_loader,
                    prototype_shape=args.prototype_shape,
                    separation_type=args.separation_type,
                    number_of_classes=args.number_of_classes,
                    class_specific=class_specific,
                    loss_coefficients=args.loss_coefficients,
                    use_bce=args.binary_cross_entropy,
                    log=logger,
                )
                save.save_model_w_condition(
                    model=ppnet,
                    model_dir=model_dir,
                    model_name=f"{epoch}-push",
                    accu=accu,
                    target_accu=0.60,
                    log=logger,
                )

                if args.prototype_activation_function != "linear":
                    tnt.last_only(model=ppnet_multi, log=logger)
                    for i in range(args.epochs_finetune):
                        logger.log_info(f"\t\t\t\titeration: \t{i}")

                        logger.csv_log_index("train_model", (fold, epoch + 1 + args.epochs_pretrain, f"last layer {i} train"))
                        _ = tnt.train(
                            model=ppnet_multi,
                            dataloader=train_loader,
                            prototype_shape=args.prototype_shape,
                            separation_type=args.separation_type,
                            number_of_classes=args.number_of_classes,
                            optimizer=last_layer_optimizer,
                            class_specific=class_specific,
                            loss_coefficients=args.loss_coefficients,
                            use_bce=args.binary_cross_entropy,
                            log=logger,
                        )

                        logger.csv_log_index("train_model", (fold, epoch + 1 + args.epochs_pretrain, f"last layer {i} validation"))
                        accu = tnt.test(
                            model=ppnet_multi,
                            dataloader=validation_loader,
                            prototype_shape=args.prototype_shape,
                            separation_type=args.separation_type,
                            number_of_classes=args.number_of_classes,
                            class_specific=class_specific,
                            loss_coefficients=args.loss_coefficients,
                            use_bce=args.binary_cross_entropy,
                            log=logger,
                        )
                        save.save_model_w_condition(
                            model=ppnet,
                            model_dir=model_dir,
                            model_name=f"{epoch}-{i}-push",
                            accu=accu,
                            target_accu=0.60,
                            log=logger,
                        )


if __name__ == "__main__":
    # python main.py --pretrained --dataset MIAS --target normal_vs_abnormal --stratified-cross-validation --grouped-cross-validation
    command_line_params = args.get_args()
    logger = Log(command_line_params.log_dir)
    try:
        args.save_args(command_line_params, logger.metadata_dir)

        config_file = args.generate_gin_config(command_line_params, logger.metadata_dir)
        exit()
        gin.parse_config_file(config_file)

        logger.create_csv_log("train_model", ("fold", "epoch", "phase"),
                              "time", "cross entropy", "cluster_loss", "separation_loss",
                              "accuracy", "micro_f1", "macro_f1", "l1", "prototype_distances")


        main(command_line_params, logger)
    except Exception as e:
        logger.log_exception(e)
