# Copyright 2020 Lorna Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
import math
import os
import time

import numpy as np
import torch
import torch.distributed
import torch.nn.parallel
import torch.optim
import torch.optim.lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from test import evaluate
from yolov4_pytorch.data import check_anchors
from yolov4_pytorch.data import check_image_size
from yolov4_pytorch.data import create_dataloader
from yolov4_pytorch.model import YOLO
from yolov4_pytorch.solver import ModelEMA
from yolov4_pytorch.utils import compute_loss
from yolov4_pytorch.utils import fitness
from yolov4_pytorch.utils import init_seeds
from yolov4_pytorch.utils import select_device

# Hyper parameters
hyper_parameters = {"lr0": 0.01,  # initial learning rate
                    "momentum": 0.937,  # SGD momentum/Adam beta1
                    "weight_decay": 5e-4,  # optimizer weight decay
                    "giou": 0.05,  # giou loss gain
                    "cls": 0.5,  # cls loss gain
                    "cls_pw": 1.0,  # cls BCELoss positive_weight
                    "obj": 1.0,  # obj loss gain (*=image_size/640 if image_size != 640)
                    "obj_pw": 1.0,  # obj BCELoss positive_weight
                    "iou_t": 0.20,  # iou training threshold
                    "anchor_t": 4.0,  # anchor-multiple threshold
                    "fl_gamma": 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
                    "hsv_h": 0.015,  # image HSV-Hue augmentation (fraction)
                    "hsv_s": 0.7,  # image HSV-Saturation augmentation (fraction)
                    "hsv_v": 0.4,  # image HSV-Value augmentation (fraction)
                    "degrees": 0.0,  # image rotation (+/- deg)
                    "translate": 0.0,  # image translation (+/- fraction)
                    "scale": 0.5,  # image scale (+/- gain)
                    "shear": 0.0}  # image shear (+/- deg)


def train():
    print(f"Hyper parameters {hyper_parameters}")
    epochs = args.epochs
    batch_size = args.batch_size
    config_file = args.config_file
    data = args.data
    weights = "weights/checkpoint.pth" if args.resume and not args.weights else args.weights
    image_size = check_image_size(args.image_size, 32)
    device = select_device(args.device, batch_size=args.batch_size)

    # Configure
    init_seeds(0)
    with open(data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)
    train_path, val_path = data_dict["train"], data_dict["val"]
    number_classes, names = int(data_dict["number_classes"]), data_dict["names"]
    assert len(names) == number_classes, f"{len(names)} names found, number_classes={number_classes} dataset in {data}"

    # Create model
    model = YOLO(config_file=config_file, number_classes=number_classes).to(device)

    # Optimizer
    accumulate = max(round(64 / batch_size), 1)  # accumulate loss before optimizing
    hyper_parameters["weight_decay"] *= batch_size * accumulate / 64  # scale weight_decay

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_parameters():
        if v.requires_grad:
            if ".bias" in k:
                pg2.append(v)  # biases
            elif ".weight" in k and ".bn" not in k:
                pg1.append(v)  # apply weight decay
            else:
                pg0.append(v)  # all else

    optimizer = torch.optim.SGD(pg0, lr=hyper_parameters["lr0"], momentum=hyper_parameters["momentum"], nesterov=True)

    optimizer.add_param_group({"params": pg1, "weight_decay": hyper_parameters["weight_decay"]})
    optimizer.add_param_group({"params": pg2})
    print(f"Optimizer groups: {len(pg2)} .bias, {len(pg1)} conv.weight, {len(pg0)} other")
    del pg0, pg1, pg2

    # Load Model
    epoch, start_epoch, best_fitness = 0, 0, 0.0
    if os.path.exists(weights):
        checkpoint = torch.load(weights, map_location=device)  # load checkpoint

        # load model
        try:
            exclude = ["anchor"]  # exclude keys
            checkpoint["state_dict"] = {k: v for k, v in checkpoint["state_dict"].items()
                                        if k in model.state_dict() and not any(x in k for x in exclude)
                                        and model.state_dict()[k].shape == v.shape}
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        except KeyError as e:
            error_msg = f"{weights} is not compatible with {config_file}. "
            error_msg += f"Specify --weights `` or specify a --config-file "
            error_msg += f"compatible with {weights}. "
            raise KeyError(error_msg) from e

        # load optimizer
        if checkpoint["optimizer"] is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
            best_fitness = checkpoint["best_fitness"]

        # load epochs
        start_epoch = checkpoint["epoch"] + 1
        if epochs < start_epoch:
            print(f"{weights} has been trained for {checkpoint['epoch']} epochs. "
                  f"Fine-tuning for {epochs} additional epochs.")
            epochs += checkpoint["epoch"]  # fine tune additional epochs

        del checkpoint

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.8 + 0.2  # cosine
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # Initialize distributed training
    if device.type != "cpu" and torch.cuda.device_count() > 1 and torch.distributed.is_available():
        torch.distributed.init_process_group(backend="nccl",  # "distributed backend"
                                             # distributed training init method
                                             init_method="tcp://127.0.0.1:18888",
                                             # number of nodes for distributed training
                                             world_size=1,
                                             # distributed training node rank
                                             rank=0)
        model = torch.nn.parallel.DistributedDataParallel(model)

    # Dataloader
    train_dataset, train_dataloader = create_dataloader(dataroot=train_path,
                                                        image_size=image_size,
                                                        batch_size=batch_size,
                                                        hyper_parameters=hyper_parameters,
                                                        augment=True,
                                                        cache=args.cache_images,
                                                        rect=False)
    _, val_dataloader = create_dataloader(dataroot=val_path,
                                          image_size=image_size,
                                          batch_size=batch_size,
                                          hyper_parameters=hyper_parameters,
                                          augment=False,
                                          cache=args.cache_images,
                                          rect=True)

    mlc = np.concatenate(train_dataset.labels, 0)[:, 0].max()  # max label class
    number_batches = len(train_dataloader)
    assert mlc < number_classes, f"Label class {mlc} exceeds number_classes={number_classes} in {args.data}. " \
                                 f"Possible class labels are 0-{number_classes - 1}"

    # Model parameters
    hyper_parameters["cls"] *= number_classes / 80.  # scale coco-tuned hyper_parameters["cls"] to current dataset
    model.number_classes = number_classes  # attach number of classes to model
    model.hyper_parameters = hyper_parameters  # attach hyper parameters to model

    # Exponential moving average
    ema = ModelEMA(model)

    # Check anchors
    check_anchors(train_dataset, model=model, thr=hyper_parameters["anchor_t"], image_size=image_size)

    # Start training
    start_time = time.time()
    pre_steps = max(3 * number_batches, 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    results = (0, 0, 0, 0, 0, 0, 0)  # "P", "R", "mAP", "F1", "val GIoU", "val Objectness", "val Classification"
    scheduler.last_epoch = start_epoch - 1  # do not move

    print(f"Image sizes {image_size} train, {image_size} test")
    print(f"Using {train_dataloader.num_workers} dataloader workers")
    print(f"Starting training for {epochs} epochs...")

    # Creates a GradScaler once at the beginning of training.
    scaler = amp.GradScaler()

    for epoch in range(start_epoch, epochs):
        model.train()

        mean_losses = torch.zeros(4, device=device)
        print("\n")
        print(f"{'Epoch':>10}{'memory':>10}{'GIoU':>10}{'obj':>10}{'cls':>10}{'total':>10}{'targets':>10}"
              f"{' image size'}")

        progress_bar = enumerate(train_dataloader)
        progress_bar = tqdm(progress_bar, total=number_batches)
        optimizer.zero_grad()
        for i, (images, targets, paths, _) in progress_bar:
            ni = i + number_batches * epoch  # number integrated batches (since train start)
            images = images.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0

            # Warm up
            if ni <= pre_steps:
                xi = [0, pre_steps]  # x interp
                accumulate = max(1, np.interp(ni, xi, [1, 64 / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x["lr"] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x["initial_lr"] * lf(epoch)])
                    if "momentum" in x:
                        x["momentum"] = np.interp(ni, xi, [0.9, hyper_parameters["momentum"]])

            # Mixed precision training
            with amp.autocast():
                outputs = model(images)
                loss, loss_items = compute_loss(outputs, targets.to(device), model)  # scaled by batch_size

            if not torch.isfinite(loss):
                print(f"WARNING: non-finite loss, ending training {loss_items}")
                return results

            # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
            # Backward passes under autocast are not recommended.
            # Backward ops run in the same dtype autocast chose for corresponding forward ops.
            scaler.scale(loss).backward()

            # Optimize
            if ni % accumulate == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                ema.update(model)

            # Print
            mean_losses = (mean_losses * i + loss_items) / (i + 1)  # update mean losses
            progress_bar.set_description(f"{epoch:>6}/{(epochs - 1):>1}{torch.cuda.memory_reserved() / 1E9:>9.1f}G"
                                         f"{mean_losses[0]:>10.4f}{mean_losses[1]:>10.4f}"
                                         f"{mean_losses[2]:>10.4f}{mean_losses[3]:>10.4f}"
                                         f"{targets.shape[0]:>10}{images.shape[-1]:>10}")

        # Scheduler
        scheduler.step()

        ema.update_attr(model)
        final_epoch = epoch + 1 == epochs
        save_json = final_epoch and args.data[-9:] == "coco2014.yaml" or args.data[-9:] == "coco2017.yaml"
        results, maps = evaluate(config_file=config_file,
                                 batch_size=batch_size,
                                 data=data,
                                 image_size=image_size,
                                 save_json=save_json,
                                 model=ema.ema.module if hasattr(ema.ema, "module") else ema.ema,
                                 dataloader=val_dataloader)

        # Tensorboard
        tags = ["train/giou_loss", "train/obj_loss", "train/cls_loss",
                "metrics/precision", "metrics/recall", "metrics/mAP_0.5", "metrics/mAP_0.5:0.95",
                "val/giou_loss", "val/obj_loss", "val/cls_loss"]
        for x, tag in zip(list(mean_losses[:-1]) + list(results), tags):
            tb_writer.add_scalar(tag, x, epoch)

        # Update best mAP
        fitness_i = fitness(np.array(results).reshape(1, -1))
        if fitness_i > best_fitness:
            best_fitness = fitness_i

        # Save model
        torch.save({"epoch": epoch,
                    "best_fitness": best_fitness,
                    "state_dict": ema.ema.module.state_dict() if hasattr(ema, "module") else ema.ema.state_dict(),
                    "optimizer": None if final_epoch else optimizer.state_dict()}, "weights/checkpoint.pth")
        if (best_fitness == fitness_i) and not final_epoch:
            torch.save({"epoch": -1,
                        "state_dict": ema.ema.module.state_dict() if hasattr(ema, "module") else ema.ema.state_dict(),
                        "optimizer": None}, "weights/model_best.pth")

    # Finish
    print(f"{epoch - start_epoch} epochs completed in {(time.time() - start_time) / 3600:.3f} hours.\n")

    torch.distributed.destroy_process_group()
    torch.cuda.empty_cache()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(usage="\n\tpython train.py --config-file configs/COCO-Detection/yolov5-small.yaml "
                                           "--data data/coco2017.yaml")
    parser.add_argument("--epochs", type=int, default=300,
                        help="500500 is YOLOv4 max batches. (default: 300)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="mini-batch size (default: 16), this is the total "
                             "batch size of all GPUs on the current node when "
                             "using Data Parallel or Distributed Data Parallel"
                             "Effective batch size is 64 // batch_size.")
    parser.add_argument("--config-file", type=str, default="configs/COCO-Detection/yolov5-small.yaml",
                        help="Neural network profile path. (default: `configs/COCO-Detection/yolov5-small.yaml`)")
    parser.add_argument("--data", type=str, default="data/coco2017.yaml",
                        help="Path to dataset. (default: data/coco2017.yaml)")
    parser.add_argument("--image-size", type=int, default=640,
                        help="Size of processing picture. (default: 640)")
    parser.add_argument("--resume", action="store_true",
                        help="resume training from `weights/checkpoint.pth`")
    parser.add_argument("--cache-images", action="store_true",
                        help="cache images for faster training.")
    parser.add_argument("--weights", type=str, default="",
                        help="Initial weights path. (default: ``)")
    parser.add_argument("--device", default="",
                        help="device id i.e. `0` or `0,1` or `cpu`. (default: ``).")
    args = parser.parse_args()
    print(args)

    # Train
    print("Start Tensorboard with `tensorboard --logdir=runs`, view at http://localhost:6006/")
    tb_writer = SummaryWriter()
    train()
