import os
import platform
import shutil
import sys
import warnings
from functools import partial

import hydra
import numpy as np
import omegaconf
import torch
from dotenv import load_dotenv
from hydra.utils import instantiate
from icecream import ic

from ProtoPNet.add_on_layers import AddOnLayers

load_dotenv()
sys.path.append(os.getenv("PROJECT_ROOT"))

import ProtoPNet.util.config_types as conf_typ
from ProtoPNet import model, push
from ProtoPNet import train_and_test as tnt
from ProtoPNet.util import helpers, save
from ProtoPNet.util.log import Log
from ProtoPNet.util.preprocess import preprocess

tick = "\u2714"
cross = "\u2718"


@hydra.main(
    version_base=None,
    config_path=os.getenv("CONFIG_PATH"),
    config_name="main_config",
)
def main(cfg: conf_typ.Config):
    logger = Log(__name__)

    try:
        warnings.showwarning = lambda message, *args: logger.exception(
            message, warn_only=True
        )

        logger.log_command_line()

        logger.create_csv_log(
            "train_model",
            ("fold", "epoch", "phase"),
            "time",
            "cross entropy",
            "cluster_loss",
            "separation_loss",
            "accuracy",
            "micro_f1",
            "macro_f1",
            "l1",
            "prototype_distances",
        )

        cfg = omegaconf.OmegaConf.to_object(cfg)

        set_seeds(cfg.seed)

        set_gpu_usage(cfg.gpu, logger)

        run_experiment(cfg, logger)
    except Exception as e:
        logger.exception(e)


def set_gpu_usage(gpu, logger):
    """
    Set the gpu usage according to the given configuration.

    :param gpu:
    :type gpu: conf_typ.Gpu
    :param logger:
    :type logger: Log
    :return:
    """
    logger.info("GPU settings")
    if not gpu.disabled:
        match platform.system():
            case "Windows" | "Linux":
                logger.info(
                    f"\t{tick if torch.cuda.is_available() else cross} available CUDA"
                )
                logger.info(
                    f"\tVisible devices set to: {os.getenv('CUDA_VISIBLE_DEVICES')}"
                )
            case "Darwin":
                logger.info(
                    f"\t{tick if torch.backends.mps.is_available() else cross} available MPS"
                )

    else:
        logger.info(f"\t{tick} disabled")


def set_seeds(seed):
    """
    Set the seeds for all used libraries to the given value.
    :param seed:
    :type seed: int | None
    :return:
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def run_experiment(cfg: conf_typ.Config, logger: Log):
    # save last commit number (source of the used code)
    commit_file = logger.metadata_dir / "commit_hash"
    commit_file.write_bytes(helpers.get_current_commit_hash())

    shutil.copy(cfg.data.set.metadata.file, logger.metadata_dir)

    model_dir = logger.log_dir / cfg.outputs.dirs.model
    model_dir.mkdir(parents=True, exist_ok=True)
    img_dir = logger.log_dir / cfg.outputs.dirs.image
    img_dir.mkdir(parents=True, exist_ok=True)

    if cfg.cross_validation.folds > 1:
        logger.info("")
        logger.info(f"{tick} cross validation")
        logger.info(f"\t{cfg.cross_validation.folds} folds")
        logger.info(
            f"\t{tick if cfg.cross_validation.stratified else cross} stratified"
        )
        logger.info(f"\t{tick if cfg.cross_validation.balanced else cross} balanced")
        logger.info(f"\t{tick if cfg.cross_validation.grouped else cross} grouped")
    else:
        logger.info(f"{cross} cross validation")

    dataset_module = instantiate(cfg.data.datamodule)

    number_of_classes = dataset_module.dataset.number_of_classes
    image_size = dataset_module.dataset.input_size
    prototype_shape = (
        cfg.prototypes.per_class * number_of_classes,
        cfg.prototypes.size,
        1,
        1,
    )
    class_specific = True

    logger.info("")
    logger.info(f"{tick if dataset_module.debug else cross} debug")

    logger.info("")
    logger.info("data settings")
    logger.info(f"\t{cfg.data.set.name}")
    logger.info(f"\t{cfg.data.set.state}")
    logger.info(f"\t{cfg.data.set.target.size}")
    logger.info(f"\t{cfg.data.set.target.name}")
    logger.info(f"\t{' x '.join(map(str, image_size))} image size")
    logger.info(f"\t{cfg.data.set.image_properties.color_channels} color channels")
    logger.info(f"\t{cfg.data.set.image_properties.std} std")
    logger.info(f"\t{cfg.data.set.image_properties.mean} mean")
    logger.info(f"\t{number_of_classes} classes")

    logger.info("")
    logger.info("prototype settings")
    logger.info(f"\t{cfg.prototypes.per_class} prototypes per class")
    logger.info(f"\t{cfg.prototypes.size} prototype size")
    logger.info(f"\t{' x '.join(map(str, prototype_shape))} prototype shape")

    logger.info("")
    logger.info("network settings")
    logger.info(f"\t{cfg.network.name} backbone")
    logger.info(f"\t{cfg.network.add_on_layer_properties.type} add on layer type")
    logger.info(
        f"\t{cfg.network.add_on_layer_properties.activation} add on activation "
        f"({AddOnLayers.get_reference(cfg.network.add_on_layer_properties.activation)})"
    )
    logger.info(f"\t{tick if cfg.network.pretrained else cross} pretrained")
    logger.info(f"\t{tick if cfg.network.backbone_only else cross} backbone only")

    train_test_parameters = {
        "prototype_shape": prototype_shape,
        "separation_type": cfg.loss.separation_type,
        "number_of_classes": number_of_classes,
        "class_specific": class_specific,
        "loss_coefficients": cfg.loss.coefficients,
        "use_bce": cfg.loss.binary_cross_entropy,
        "backbone_only": cfg.network.backbone_only,
        "log": logger,
        "device": cfg.gpu.device,
    }
    partial_train = partial(tnt.train, **train_test_parameters)
    partial_test = partial(tnt.test, **train_test_parameters)

    preprocess_parameters = {
        "mean": cfg.data.set.image_properties.mean,
        "std": cfg.data.set.image_properties.std,
        "number_of_channels": cfg.data.set.image_properties.color_channels,
    }
    partial_preprocess = partial(preprocess, **preprocess_parameters)

    # train the model
    logger.info("start training")

    for fold, (train_sampler, validation_sampler) in enumerate(
        dataset_module.folds, start=1
    ):
        # construct the model
        ppnet = model.construct_PPNet(
            base_architecture=cfg.network.name,
            pretrained=cfg.network.pretrained,
            color_channels=cfg.data.set.image_properties.color_channels,
            img_shape=image_size,
            prototype_shape=prototype_shape,
            num_classes=number_of_classes,
            prototype_activation_function=cfg.prototypes.activation_fn,
            add_on_layers_type=cfg.network.add_on_layer_properties.type,
            backbone_only=cfg.network.backbone_only,
            positive_weights_in_classifier=False,
        )
        if not cfg.gpu.disabled:
            ppnet = ppnet.to(cfg.gpu.device)

        if fold == 1:
            logger.info(f"\n{ppnet}\n")

        ppnet_multi = torch.nn.DataParallel(ppnet)

        warm_optimizer_specs = [
            {
                "params": ppnet.add_on_layers.parameters(),
                "lr": cfg.phases.warm.learning_rates["add_on_layers"],
                "weight_decay": cfg.phases.warm.weight_decay,
            },
        ]
        if not cfg.network.backbone_only:
            warm_optimizer_specs += [
                {
                    "params": ppnet.prototype_vectors,
                    "lr": cfg.phases.warm.learning_rates["prototype_vectors"],
                },
            ]
        warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

        joint_optimizer_specs = [
            {
                "params": ppnet.features.parameters(),
                "lr": cfg.phases.joint.learning_rates["features"],
                "weight_decay": cfg.phases.joint.weight_decay,
            },  # bias are now also being regularized
            {
                "params": ppnet.add_on_layers.parameters(),
                "lr": cfg.phases.joint.learning_rates["add_on_layers"],
                "weight_decay": cfg.phases.joint.weight_decay,
            },
        ]
        if not cfg.network.backbone_only:
            joint_optimizer_specs += [
                {
                    "params": ppnet.prototype_vectors,
                    "lr": cfg.phases.joint.learning_rates["prototype_vectors"],
                },
            ]

        joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
        joint_lr_scheduler = instantiate(cfg.phases.joint.scheduler)(joint_optimizer)

        last_layer_optimizer_specs = [
            {
                "params": ppnet.last_layer.parameters(),
                "lr": cfg.phases.finetune.learning_rates["features"],
            }
        ]
        last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

        dataset_module.log_data_information(logger)

        if not cfg.network.backbone_only and cfg.phases.warm.epochs > 0:
            tnt.warm_only(
                model=ppnet_multi, log=logger, backbone_only=cfg.network.backbone_only
            )

            train_loader = dataset_module.train_dataloader(
                sampler=train_sampler,
                batch_size=cfg.phases.warm.batch_size,
            )
            validation_loader = dataset_module.validation_dataloader(
                sampler=validation_sampler,
                batch_size=cfg.phases.warm.batch_size,
            )

            logger.info(f"\tbatch size: {train_loader.batch_size}")
            logger.info(f"\t\ttrain: {train_loader.batch_size}")
            logger.info(f"\t\tvalidation: {validation_loader.batch_size}")

            for epoch in np.arange(cfg.phases.warm.epochs) + 1:
                logger.info(f"\t\twarm epoch: \t{epoch}")
                logger.csv_log_index("train_model", (fold, epoch, "warm train"))
                _ = partial_train(
                    model=ppnet_multi,
                    dataloader=train_loader,
                    optimizer=warm_optimizer,
                )

                logger.csv_log_index("train_model", (fold, epoch, "warm validation"))
                accu = partial_test(
                    model=ppnet_multi,
                    dataloader=validation_loader,
                )
                save.save_model_w_condition(
                    model=ppnet,
                    model_dir=model_dir,
                    model_name=f"{epoch}-warm",
                    accu=accu,
                    target_accu=0.60,
                    log=logger,
                )

            logger.info("\t\tfinished warmup")
        else:
            cfg.phases.warm.epochs = 0

        tnt.joint(
            model=ppnet_multi, log=logger, backbone_only=cfg.network.backbone_only
        )

        train_loader = dataset_module.train_dataloader(
            sampler=train_sampler,
            batch_size=cfg.phases.joint.batch_size,
        )
        validation_loader = dataset_module.validation_dataloader(
            sampler=validation_sampler,
            batch_size=cfg.phases.joint.batch_size,
        )
        push_loader = dataset_module.push_dataloader(
            sampler=train_sampler,
            batch_size=cfg.phases.push.batch_size,
        )

        logger.info("\t\tbatch size:")
        logger.info(f"\t\t\ttrain: {train_loader.batch_size}")
        logger.info(f"\t\t\tvalidation: {validation_loader.batch_size}")
        logger.info(f"\t\t\tpush: {push_loader.batch_size}")

        for epoch in np.arange(cfg.phases.joint.epochs) + 1:
            real_epoch_number = epoch + cfg.phases.warm.epochs
            logger.info(f"\t\tepoch: \t{epoch} ({real_epoch_number})")
            if epoch > 1:
                joint_lr_scheduler.step()

            logger.csv_log_index("train_model", (fold, real_epoch_number, "train"))
            _ = partial_train(
                model=ppnet_multi,
                dataloader=train_loader,
                optimizer=joint_optimizer,
            )

            logger.csv_log_index("train_model", (fold, real_epoch_number, "validation"))
            accu = partial_test(
                model=ppnet_multi,
                dataloader=validation_loader,
            )
            save.save_model_w_condition(
                model=ppnet,
                model_dir=model_dir,
                model_name=f"{real_epoch_number}-no_push",
                accu=accu,
                target_accu=0.60,
                log=logger,
            )

            if not cfg.network.backbone_only and epoch in cfg.phases.push.push_epochs:
                push.push_prototypes(
                    push_loader,  # pytorch dataloader (must be unnormalized in [0,1])
                    prototype_network_parallel=ppnet_multi,
                    # pytorch network with prototype_vectors
                    class_specific=class_specific,
                    preprocess_input_function=partial_preprocess,  # normalize
                    prototype_layer_stride=1,
                    root_dir_for_saving_prototypes=img_dir,
                    # if not None, prototypes will be saved here
                    epoch_number=epoch,
                    # if not provided, prototypes saved previously will be overwritten
                    prototype_img_filename_prefix=cfg.outputs.file_prefixes.prototype,
                    prototype_self_act_filename_prefix=cfg.outputs.file_prefixes.self_activation,
                    proto_bound_boxes_filename_prefix=cfg.outputs.file_prefixes.bounding_box,
                    save_prototype_class_identity=True,
                    log=logger,
                    device=cfg.gpu.device,
                )

                logger.csv_log_index(
                    "train_model", (fold, real_epoch_number, "push validation")
                )
                accu = partial_test(
                    model=ppnet_multi,
                    dataloader=validation_loader,
                )
                save.save_model_w_condition(
                    model=ppnet,
                    model_dir=model_dir,
                    model_name=f"{real_epoch_number}-push",
                    accu=accu,
                    target_accu=0.60,
                    log=logger,
                )

                if cfg.prototypes.activation_fn != "linear":
                    tnt.last_only(model=ppnet_multi, log=logger)
                    for i in np.arange(cfg.phases.finetune.epochs) + 1:
                        logger.info(f"\t\t\t\titeration: \t{i}")

                        logger.csv_log_index(
                            "train_model",
                            (fold, real_epoch_number, f"last layer {i} train"),
                        )
                        _ = partial_train(
                            model=ppnet_multi,
                            dataloader=train_loader,
                            optimizer=last_layer_optimizer,
                        )

                        logger.csv_log_index(
                            "train_model",
                            (fold, real_epoch_number, f"last layer {i} validation"),
                        )
                        accu = partial_test(
                            model=ppnet_multi,
                            dataloader=validation_loader,
                        )
                        save.save_model_w_condition(
                            model=ppnet,
                            model_dir=model_dir,
                            model_name=f"{real_epoch_number}-{i}-push",
                            accu=accu,
                            target_accu=0.60,
                            log=logger,
                        )
                    logger.info("\t\t\t\tfinished finetuning last layer")
                logger.info("\t\t\tfinished pushing prototypes")
        logger.info(f"\t\tfinished training fold {fold}")

    logger.info("finished training")


if __name__ == "__main__":
    conf_typ.init_config_store()

    main()
