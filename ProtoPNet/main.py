import gin
import numpy as np
import os
import re
import shutil
import sys
import torch
import warnings

from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
sys.path.append(os.getenv("PROJECT_ROOT"))

from ProtoPNet import model
from ProtoPNet import push
from ProtoPNet import train_and_test as tnt

from ProtoPNet.dataset.metadata import DATASETS
# needed to configure dataset using gin
from ProtoPNet.dataset.dataloaders.MIAS import MIASDataModule
from ProtoPNet.dataset.dataloaders.DDSM import DDSMDataModule

from ProtoPNet.util import args as ProtoPNet_args
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
    log_source_dir = args.log_dir / "src"
    log_source_dir.mkdir(parents=True, exist_ok=True)

    dataset_information = DATASETS[args.dataset]

    shutil.copy(dataset_information.METADATA.FILE, logger.metadata_dir)

    # copy code to result directory
    base_architecture_type = re.match("^[a-z]*", args.backbone).group(0)

    project_root = Path(os.getenv("PROJECT_ROOT"))
    module_root = project_root / "ProtoPNet"
    shutil.copy(
        src=module_root / "config" / "backbone_features" / f"{base_architecture_type}_features.py",
        dst=log_source_dir,
    )
    shutil.copy(
        src=module_root / "model.py",
        dst=log_source_dir
    )
    shutil.copy(
        src=module_root / "train_and_test.py",
        dst=log_source_dir
    )

    model_dir = args.log_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    img_dir = args.log_dir / "img"
    img_dir.mkdir(parents=True, exist_ok=True)

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

            for epoch in np.arange(args.epochs_pretrain) + 1:
                logger.log_info(f"\t\twarm epoch: \t{epoch}")
                logger.csv_log_index("train_model", (fold, epoch, "warm train"))
                _ = tnt.train(
                    model=ppnet_multi,
                    dataloader=train_loader,
                    optimizer=warm_optimizer,
                    log=logger,
                )

                logger.csv_log_index("train_model", (fold, epoch, "warm validation"))
                accu = tnt.test(
                    model=ppnet_multi,
                    dataloader=validation_loader,
                    log=logger,
                )
                save.save_model_w_condition(
                    model=ppnet,
                    model_dir=model_dir,
                    model_name=f"{epoch}-warm",
                    accu=accu,
                    target_accu=0.60,
                    log=logger,
                )

            logger.log_info("\t\tfinished warmup")

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

        for epoch in np.arange(args.epochs) + 1:
            real_epoch_number = epoch + args.epochs_pretrain
            logger.log_info(f"\t\tepoch: \t{epoch} ({real_epoch_number})")
            if epoch > 1:
                joint_lr_scheduler.step()

            logger.csv_log_index("train_model", (fold, real_epoch_number, "train"))
            _ = tnt.train(
                model=ppnet_multi,
                dataloader=train_loader,
                optimizer=joint_optimizer,
                log=logger,
            )

            logger.csv_log_index("train_model", (fold, real_epoch_number, "validation"))
            accu = tnt.test(
                model=ppnet_multi,
                dataloader=validation_loader,
                log=logger,
            )
            save.save_model_w_condition(
                model=ppnet,
                model_dir=model_dir,
                model_name=f"{real_epoch_number}-no_push",
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

                logger.csv_log_index("train_model", (fold, real_epoch_number, "push validation"))
                accu = tnt.test(
                    model=ppnet_multi,
                    dataloader=validation_loader,
                    log=logger,
                )
                save.save_model_w_condition(
                    model=ppnet,
                    model_dir=model_dir,
                    model_name=f"{real_epoch_number}-push",
                    accu=accu,
                    target_accu=0.60,
                    log=logger,
                )

                if args.prototype_activation_function != "linear":
                    tnt.last_only(model=ppnet_multi, log=logger)
                    for i in np.arange(args.epochs_finetune) + 1:
                        logger.log_info(f"\t\t\t\titeration: \t{i}")

                        logger.csv_log_index("train_model", (fold, real_epoch_number, f"last layer {i} train"))
                        _ = tnt.train(
                            model=ppnet_multi,
                            dataloader=train_loader,
                            optimizer=last_layer_optimizer,
                            log=logger,
                        )

                        logger.csv_log_index("train_model", (fold, real_epoch_number, f"last layer {i} validation"))
                        accu = tnt.test(
                            model=ppnet_multi,
                            dataloader=validation_loader,
                            log=logger,
                        )
                        save.save_model_w_condition(
                            model=ppnet,
                            model_dir=model_dir,
                            model_name=f"{real_epoch_number}-{i}-push",
                            accu=accu,
                            target_accu=0.60,
                            log=logger,
                        )
                    logger.log_info("\t\t\t\tfinished finetuning last layer")
                logger.log_info("\t\t\tfinished pushing prototypes")
        logger.log_info(f"\t\tfinished training fold {fold}")

    logger.log_info("finished training")


if __name__ == "__main__":
    # python main.py --pretrained --batch-size-pretrain 32 --batch-size 8 --batch-size-push 16 --epochs-pretrain 4 --epochs-finetune 4 --epochs 10 --dataset MIAS --target normal_vs_abnormal --stratified-cross-validation --grouped-cross-validation --gpu-id 1 --log-dir original-n-v-a
    command_line_params = ProtoPNet_args.get_args()
    logger = Log(command_line_params.log_dir)

    try:
        warnings.showwarning = lambda warning: logger.log_exception(warning, warn_only=True)

        logger.log_command_line()
        ProtoPNet_args.save_args(command_line_params, logger.metadata_dir)

        config_file = ProtoPNet_args.generate_gin_config(command_line_params, logger.metadata_dir)
        gin.parse_config_file(config_file)
        exit()

        logger.create_csv_log("train_model", ("fold", "epoch", "phase"),
                              "time", "cross entropy", "cluster_loss", "separation_loss",
                              "accuracy", "micro_f1", "macro_f1", "l1", "prototype_distances")

        main(command_line_params, logger)
    except Exception as e:
        logger.log_exception(e)
