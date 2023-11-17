import argparse
import gin
import pickle
# import os
import shutil
import torch

from dotenv import load_dotenv
from pathlib import Path
load_dotenv()

from ProtoPNet.dataset.dataloaders import CustomDataModule
import ProtoPNet.prune as prune
import ProtoPNet.train_and_test as tnt
from ProtoPNet.util.log import Log
from ProtoPNet.util.preprocess import preprocess
import ProtoPNet.util.save as save

parser = argparse.ArgumentParser()
parser.add_argument("--model-dir", type=str, required=True)
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--gpu-id", type=str, default="0")

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

optimize_last_layer = True

# pruning parameters
k = 6
prune_threshold = 3

original_model_dir = Path(args.model_dir)  # './runs/run-name/models/'
original_model_name = args.model     # '10_16push0.8007.pth'
original_model_path = original_model_dir / original_model_name

# pruning must happen after push
assert "no_push" not in original_model_name, "pruning must happen after push"

logger = Log(
    log_dir=original_model_dir,
    log_file="log-prune.txt",
)

gin.parse_config_file(logger.metadata_dir / "train.gin")

epoch = original_model_name.split("-push")[0]

if "-" in epoch:
    epoch = int(epoch.split("-")[0])
else:
    epoch = int(epoch)

model_dir = original_model_dir / f"pruned_prototypes_epoch{epoch}_k{k}_pt{prune_threshold}"
model_dir.mkdir(parents=True, exist_ok=True)
shutil.copy(src=Path.cwd() / __file__, dst=model_dir)

ppnet = torch.load(original_model_path)
ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)
class_specific = True

train_args = pickle.load((logger.metadata_dir / "args.pickle").open(mode="rb"))

data_module = CustomDataModule(
    train_args.dataset_config,
    train_args.used_images,
    train_args.target,
    num_workers=train_args.number_of_workers,
    seed=train_args.seed
)

train_batch_size = 80
test_batch_size = 100
train_push_batch_size = 80

# train set
train_loader = data_module.train_dataloader(train_batch_size)

# test set
valid_loader = data_module.test_dataloader(test_batch_size)

logger(f"INFO: training set size: {len(train_loader.dataset)}")
logger(f"INFO: test set size: {len(valid_loader.dataset)}")
logger(f"INFO: batch size: {train_batch_size}")

# push set: needed for pruning because it is unnormalized
train_push_loader = data_module.push_dataloader(train_push_batch_size)

logger(f"INFO: push set size: {len(train_push_loader.dataset)}")

tnt.test(
    model=ppnet_multi,
    dataloader=valid_loader,
    log=logger,
)

# prune prototypes
logger("INFO: prune")
prune.prune_prototypes(
    dataloader=train_push_loader,
    prototype_network_parallel=ppnet_multi,
    k=k,
    prune_threshold=prune_threshold,
    preprocess_input_function=preprocess,  # normalize
    original_model_dir=original_model_dir,
    epoch_number=epoch,
    # model_name=None,
    log=logger,
    copy_prototype_imgs=True,
)
accu = tnt.test(
    model=ppnet_multi,
    dataloader=valid_loader,
    log=logger,
)
save.save_model_w_condition(
    model=ppnet,
    model_dir=model_dir,
    model_name=original_model_name.split("push")[0] + "prune",
    accu=accu,
    target_accu=0.70,
    log=logger,
)

# last layer optimization
if optimize_last_layer:
    last_layer_optimizer_specs = [
        {"params": ppnet.last_layer.parameters(), "lr": 1e-4}
    ]
    last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

    logger("INFO: optimize last layer")
    tnt.last_only(model=ppnet_multi, log=logger)
    for i in range(100):
        logger(f"INFO:\titeration: \t{i}")
        _ = tnt.train(
            model=ppnet_multi,
            dataloader=train_loader,
            optimizer=last_layer_optimizer,
            log=logger,
        )
        accu = tnt.test(
            model=ppnet_multi,
            dataloader=valid_loader,
            log=logger,
        )
        save.save_model_w_condition(
            model=ppnet,
            model_dir=model_dir,
            model_name=original_model_name.split("push")[0]
            + "_"
            + str(i)
            + "prune",
            accu=accu,
            target_accu=0.70,
            log=logger,
        )
