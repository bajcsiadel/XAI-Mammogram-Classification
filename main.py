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
from ProtoPNet.dataset.dataloaders.MIAS import MIASDataModule
from ProtoPNet.dataset.dataloaders.DDSM import DDSMDataModule

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
    torch.cuda.set_device(args.gpu_id)
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
        logger.log_info("using cross-validation with:"
                        f"\t{args.cross_validation_folds} folds")
        if args.stratified_cross_validation:
            logger.log_info(f"\tstratified")
        if args.grouped_cross_validation:
            logger.log_info(f"\tgrouped")

    full_train_loader = dataset_module.train_dataloader()
    push_loader = dataset_module.push_dataloader()
    test_loader = dataset_module.test_dataloader()

    logger.log_info(f"training set size: {len(full_train_loader)}")
    logger.log_info(f"push set size: {len(push_loader)}")
    logger.log_info(f"test set size: {len(test_loader)}")
    logger.log_info(f"batch size: {args.batch_size}")
    logger.log_info(f"number of prototypes per class: {args.prototypes_per_class}")

    # construct the model
    # parameters will set from the gin config file
    ppnet = model.construct_PPNet()
    # if prototype_activation_function == 'linear':
    #    ppnet.set_last_layer_incorrect_connection(incorrect_strength=0)
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

    # train the model
    logger.log_info("start training")

    for fold, (train_loader, validation_loader) in full_train_loader:
        logger.log_info(f"\tFOLD {fold + 1}")
        logger.log_info(f"\t\ttrain set size: {len(train_loader)}")
        logger.log_info(f"\t\tvalidation set size: {len(validation_loader)}")

        for epoch in range(args.epochs):
            logger.log_info("\t\tepoch: \t{0}".format(epoch))

            if not args.backbone_only and epoch < args.epochs_pretrain:
                tnt.warm_only(
                    model=ppnet_multi, log=logger, backbone_only=args.backbone_only
                )
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
            else:
                tnt.joint(model=ppnet_multi, log=logger, backbone_only=args.backbone_only)
                if epoch > 0:
                    joint_lr_scheduler.step()
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
                model_name=str(epoch) + "_nopush",
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
                    model_name=str(epoch) + "_push",
                    accu=accu,
                    target_accu=0.60,
                    log=logger,
                )

                if args.prototype_activation_function != "linear":
                    tnt.last_only(model=ppnet_multi, log=logger)
                    for i in range(args.epochs_finetune):
                        logger.log_info(f"iteration: \t{i}")
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
                            model_name=str(epoch) + "_" + str(i) + "_push",
                            accu=accu,
                            target_accu=0.60,
                            log=logger,
                        )


if __name__ == "__main__":
    # python main.py --pretrained --dataset MIAS --target normal_vs_abnormal --stratified-cross-validation --grouped-cross-validation
    command_line_params = args.get_args()
    logger = Log(command_line_params.log_dir)
    try:
        config_file = args.generate_gin_config(command_line_params, logger.metadata_dir)
        gin.parse_config_file(config_file)

        args.save_args(command_line_params, logger.metadata_dir)

        main(command_line_params, logger)
    except Exception as e:
        logger.log_exception(e)
