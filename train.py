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
import glob
import math
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from test import evaluate
from yolov4_pytorch.data import LoadImagesAndLabels
from yolov4_pytorch.data import check_image_size
from yolov4_pytorch.model import YOLO
from yolov4_pytorch.solver import ModelEMA
from yolov4_pytorch.utils import compute_loss
from yolov4_pytorch.utils import fitness
from yolov4_pytorch.utils import init_seeds
from yolov4_pytorch.utils import labels_to_class_weights
from yolov4_pytorch.utils import labels_to_image_weights
from yolov4_pytorch.utils import plot_images
from yolov4_pytorch.utils import plot_labels
from yolov4_pytorch.utils import plot_results
from yolov4_pytorch.utils import print_mutation
from yolov4_pytorch.utils import select_device

mixed_precision = True
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    print("Apex recommended for faster mixed precision training: https://github.com/NVIDIA/apex")
    mixed_precision = False  # not installed

# Hyper parameters
hyper_parameters = {"lr0": 0.01,  # initial learning rate (SGD=1E-2, Adam=1E-3)
                    "momentum": 0.937,  # SGD momentum
                    "weight_decay": 5e-4,  # optimizer weight decay
                    "giou": 0.05,  # giou loss gain
                    "classes": 0.58,  # cls loss gain
                    "classes_pw": 1.0,  # cls BCELoss positive_weight
                    "obj": 1.0,  # obj loss gain (*=image_size/320 if image_size != 320)
                    "obj_pw": 1.0,  # obj BCELoss positive_weight
                    "iou_t": 0.20,  # iou training threshold
                    "anchor_t": 4.0,  # anchor-multiple threshold
                    "fl_gamma": 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
                    "hsv_h": 0.014,  # image HSV-Hue augmentation (fraction)
                    "hsv_s": 0.68,  # image HSV-Saturation augmentation (fraction)
                    "hsv_v": 0.36,  # image HSV-Value augmentation (fraction)
                    "degrees": 0.0,  # image rotation (+/- deg)
                    "translate": 0.0,  # image translation (+/- fraction)
                    "scale": 0.5,  # image scale (+/- gain)
                    "shear": 0.0}  # image shear (+/- deg)
print(hyper_parameters)

# Overwrite hyp with hyp*.txt
parameter_file = glob.glob("hyp*.txt")
if parameter_file:
    print(f"Using {parameter_file[0]}")
    for keys, value in zip(hyper_parameters.keys(), np.loadtxt(parameter_file[0])):
        hyper_parameters[keys] = value

# Print focal loss if gamma > 0
if hyper_parameters["fl_gamma"]:
    print(f"Using FocalLoss(gamma={hyper_parameters['fl_gamma']})")


def train(parameters):
    epochs = args.epochs  # 300
    batch_size = args.batch_size  # 64
    weights = args.weights  # initial training weights

    # Configure
    init_seeds(1)
    with open(args.data) as data_file:
        data_dict = yaml.load(data_file, Loader=yaml.FullLoader)  # model dict
    train_path = data_dict["train"]
    test_path = data_dict["val"]
    classes = 1 if args.single_cls else int(data_dict["classes"])  # number of classes

    # Remove previous results
    for old_file in glob.glob("*_batch_*.png") + glob.glob("result.txt"):
        os.remove(old_file)

    # Create model
    model = YOLO(args.config_file).to(device)
    assert model.config_file[
               "classes"] == classes, f"{args.data} nc={classes} classes but {args.config_file} classes={model.config_file['classes']} classes "

    # Image sizes
    gs = int(max(model.stride))  # grid size (max stride)
    image_size, image_size_test = [check_image_size(size, gs) for size in args.image_size]

    # Optimizer
    nominal_batch_size = 64  # nominal batch size
    accumulate = max(round(nominal_batch_size / batch_size), 1)  # accumulate loss before optimizing
    parameters["weight_decay"] *= batch_size * accumulate / nominal_batch_size  # scale weight_decay
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for model_key, model_value in model.named_parameters():
        if model_value.requires_grad:
            if ".bias" in model_key:
                pg2.append(model_value)  # biases
            elif ".weight" in model_key and ".bn" not in model_key:
                pg1.append(model_value)  # apply weight decay
            else:
                pg0.append(model_value)  # all else

    optimizer = optim.SGD(pg0, lr=parameters["lr0"], momentum=parameters["momentum"], nesterov=True)
    optimizer.add_param_group({"params": pg1, "weight_decay": parameters["weight_decay"]})  # add pg1 with weight_decay
    optimizer.add_param_group({"params": pg2})  # add pg2 (biases)
    print(f"Optimizer groups: {len(pg2)} .bias, {len(pg1)} conv.weight, {len(pg0)} other")
    del pg0, pg1, pg2

    # Load Model
    start_epoch, best_fitness = 0, 0.0

    if weights.endswith(".pth"):
        checkpoint = torch.load(weights, map_location=device)  # load checkpoint

        # load model
        try:
            checkpoint["state_dict"] = {k: v for k, v in checkpoint["state_dict"].float().items() if
                                        model.state_dict()[k].shape == v.shape}
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        except KeyError as e:
            s = f"{args.weights} is not compatible with {args.config_file}. Specify --weights "" or specify a " \
                f"--config-file compatible with {args.weights}. "
            raise KeyError(s) from e

        # load optimizer
        if checkpoint["optimizer"] is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
            best_fitness = checkpoint["best_fitness"]

        # load results
        if checkpoint.get("training_results") is not None:
            with open("results.txt", "w") as file:
                file.write(checkpoint["training_results"])  # write results.txt

        start_epoch = checkpoint["epoch"] + 1
        del checkpoint

    # Mixed precision training https://github.com/NVIDIA/apex
    if mixed_precision:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lr_lambda = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.9 + 0.1  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scheduler.last_epoch = start_epoch - 1  # do not move

    # Initialize distributed training
    if device.type != "cpu" and torch.cuda.device_count() > 1 and torch.distributed.is_available():
        dist.init_process_group(backend="nccl",  # distributed backend
                                init_method="tcp://127.0.0.1:18888",  # init method
                                world_size=1,  # number of nodes
                                rank=0)  # node rank
        model = torch.nn.parallel.DistributedDataParallel(model)

    # Dataset
    # Apply augmentation hyper parameters (option: rectangular training)
    train_dataset = LoadImagesAndLabels(train_path, image_size, batch_size,
                                        augment=True,
                                        hyper_parameters=parameters,  # augmentation hyper parameters
                                        rect=args.rect,  # rectangular training
                                        cache_images=args.cache_images,
                                        single_cls=args.single_cls)
    test_dataset = LoadImagesAndLabels(test_path, image_size_test, batch_size,
                                       hyper_parameters=parameters,
                                       rect=True,
                                       cache_images=args.cache_images,
                                       single_cls=args.single_cls)
    collate_fn = train_dataset.collate_fn

    max_class = np.concatenate(train_dataset.labels, 0)[:, 0].max()
    assert max_class < classes, f"Label class {max_class} exceeds classes={classes} in {args.config_file}. Correct your labels or your model."

    # Dataloader
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   num_workers=args.workers,
                                                   shuffle=not args.rect,
                                                   # Shuffle=True unless rectangular training is used
                                                   pin_memory=True,
                                                   collate_fn=collate_fn)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size,
                                                  num_workers=args.workers,
                                                  pin_memory=True,
                                                  collate_fn=collate_fn)

    # Model parameters
    parameters["classes"] *= classes / 80.  # scale COCO-Detection-tuned parameters["classes"] to current dataset
    model.classes = classes  # attach number of classes to model
    model.hyper_parameters = parameters  # attach hyper parameters to model
    model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
    model.class_weights = labels_to_class_weights(train_dataset.labels, classes).to(device)  # attach class weights
    model.names = data_dict["names"]

    # class frequency
    labels = np.concatenate(train_dataset.labels, 0)
    c = torch.tensor(labels[:, 0])  # classes
    plot_labels(labels)
    tb_writer.add_histogram("classes", c, 0)

    # Exponential moving average
    ema = ModelEMA(model)

    # Start training
    t0 = time.time()
    nb = len(train_dataloader)  # number of batches
    n_burn = max(3 * nb, 1000)  # burn-in iterations, max(3 epochs, 1k iterations)
    maps = np.zeros(classes)  # mAP per class
    # "P", "R", "mAP", "F1", "val GIoU", "val Objectness", "val Classification"
    results = (0, 0, 0, 0, 0, 0, 0)
    print(f"Image sizes {image_size} train, {image_size_test} test")
    print(f"Using {args.workers} dataloader workers")
    print(f"Starting training for {epochs} epochs...")
    epoch = 0
    for epoch in range(start_epoch, epochs):
        model.train()

        # Update image weights (optional)
        if train_dataset.image_weights:
            w = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
            image_weights = labels_to_image_weights(train_dataset.labels, num_classes=classes, class_weights=w)
            train_dataset.indices = random.choices(range(train_dataset.image_files_num), weights=image_weights,
                                                   k=train_dataset.image_files_num)  # rand weighted idx

        mean_losses = torch.zeros(4, device=device)  # mean losses
        print(("\n" + "%10s" * 8) % ("Epoch", "memory", "GIoU", "obj", "cls", "total", "targets", " image_size"))
        progress_bar = tqdm(enumerate(train_dataloader), total=nb)  # progress bar
        ni = 0
        for index, (images, targets, paths, _) in progress_bar:
            ni = index + nb * epoch  # number integrated batches (since train start)
            images = images.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0

            # Burn-in
            if ni <= n_burn:
                xi = [0, n_burn]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
                accumulate = max(1, np.interp(ni, xi, [1, nominal_batch_size / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x["lr"] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x["initial_lr"] * lr_lambda(epoch)])
                    if "momentum" in x:
                        x["momentum"] = np.interp(ni, xi, [0.9, parameters["momentum"]])

            # Multi-scale
            if args.multi_scale:
                sz = random.randrange(image_size * 0.5, image_size * 1.5 + gs) // gs * gs  # size
                sf = sz / max(images.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(shape * sf / gs) * gs for shape in images.shape[2:]]
                    images = F.interpolate(images, size=ns, mode="bilinear", align_corners=False)

            # Forward
            output = model(images)

            # Loss
            loss, loss_items = compute_loss(output, targets.to(device), model)
            if not torch.isfinite(loss):
                print("WARNING: non-finite loss, ending training ", loss_items)
                return 0, 0, 0, 0, 0, 0, 0

            # Backward
            if mixed_precision:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # Optimize
            if ni % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()
                ema.update(model)

            # Print
            mean_losses = (mean_losses * index + loss_items) / (index + 1)  # update mean losses
            memory = "%.3gG" % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ("%10s" * 2 + "%10.4g" * 6) % (
                "%g/%g" % (epoch, epochs - 1), memory, *mean_losses, targets.shape[0], images.shape[-1])
            progress_bar.set_description(s)

            # Plot
            if ni < 3:
                filename = f"train_batch_{index}.png"
                res = plot_images(images=images, targets=targets, paths=paths, filename=filename)
                if tb_writer:
                    tb_writer.add_image(filename, res, dataformats="HWC", global_step=epoch)

        # Scheduler
        scheduler.step()

        # mAP
        ema.update_attr(model)
        final_epoch = epoch + 1 == epochs
        if not args.notest or final_epoch:  # Calculate mAP
            results, maps, times = evaluate(args.config_file,
                                            args.data,
                                            batch_size=batch_size,
                                            image_size=image_size_test,
                                            save_json=final_epoch,
                                            model=ema.ema,
                                            single_cls=args.single_cls,
                                            dataloader=test_dataloader,
                                            fast=ni < n_burn)

        # Write
        with open("results.txt", "a") as f:
            f.write(s + "%10.4g" * 7 % results + "\n")  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)

        # Tensorboard
        if tb_writer:
            tags = ["train/giou_loss", "train/obj_loss", "train/cls_loss",
                    "metrics/precision", "metrics/recall", "metrics/mAP_0.5", "metrics/F1",
                    "val/giou_loss", "val/obj_loss", "val/cls_loss"]
            for scalar_value, tag in zip(list(mean_losses[:-1]) + list(results), tags):
                tb_writer.add_scalar(tag, scalar_value, epoch)

        # Update best mAP
        fi = fitness(np.array(results).reshape(1, -1))  # fitness_i = weighted combination of [P, R, mAP, F1]
        if fi > best_fitness:
            best_fitness = fi

        # Save model
        save = (not args.nosave) or (final_epoch and not args.evolve)
        if save:
            with open("results.txt", "r") as f:  # create checkpoint
                state = {"epoch": epoch,
                         "best_fitness": best_fitness,
                         "training_results": f.read(),
                         "state_dict": ema.ema.module.state_dict() if hasattr(model,
                                                                              "module") else ema.ema.state_dict(),
                         "optimizer": None if final_epoch else optimizer.state_dict()}

            # Save last, best and delete
            torch.save(state, "weights/checkpoint.pth")
            if (best_fitness == fi) and not final_epoch:
                state = {"epoch": -1,
                         "best_fitness": None,
                         "training_results": None,
                         "state_dict": ema.ema.module.state_dict() if hasattr(model,
                                                                              "module") else ema.ema.state_dict(),
                         "optimizer": None}
                torch.save(state, "weights/model_best.pth")
            del state

    if not args.evolve:
        plot_results()  # save as results.png
    print(f"{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.\n")
    dist.destroy_process_group() if torch.cuda.device_count() > 1 else None
    torch.cuda.empty_cache()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300,
                        help="500500 is YOLOv4 max batches. (default: 300)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="mini-batch size (default: 16), this is the total "
                             "batch size of all GPUs on the current node when "
                             "using Data Parallel or Distributed Data Parallel"
                             "Effective batch size is 64 // batch_size.")
    parser.add_argument("--config-file", type=str, default="configs/COCO-Detection/yolov5s.yaml",
                        help="Neural network profile path. (default: `configs/COCO-Detection/yolov5s.yaml`)")
    parser.add_argument("--data", type=str, default="data/coco2014.yaml",
                        help="Path to dataset. (default: data/coco2014.yaml)")
    parser.add_argument("--workers", default=8, type=int, metavar="N",
                        help="Number of data loading workers (default: 8)")
    parser.add_argument("--multi-scale", action="store_true",
                        help="adjust (67%% - 150%%) img_size every 10 batches")
    parser.add_argument("--image-size", nargs="+", type=int, default=[640, 640],
                        help="Size of processing picture. (default: [640, 640])")
    parser.add_argument("--rect", action="store_true",
                        help="rectangular training for faster training.")
    parser.add_argument("--resume", action="store_true",
                        help="resume training from checkpoint.pth")
    parser.add_argument("--nosave", action="store_true",
                        help="only save final checkpoint")
    parser.add_argument("--notest", action="store_true",
                        help="only test final epoch")
    parser.add_argument("--evolve", action="store_true",
                        help="evolve hyper parameters")
    parser.add_argument("--cache-images", action="store_true",
                        help="cache images for faster training.")
    parser.add_argument("--weights", type=str, default="",
                        help="Initial weights path. (default: ``)")
    parser.add_argument("--device", default="",
                        help="device id (i.e. 0 or 0,1 or cpu)")
    parser.add_argument("--single-cls", action="store_true",
                        help="train as single-class dataset")
    args = parser.parse_args()

    args.weights = "weights/checkpoint.pth" if args.resume else args.weights

    print(args)
    args.image_size.extend([args.image_size[-1]] * (2 - len(args.image_size)))  # extend to 2 sizes (train, test)
    device = select_device(args.device, apex=mixed_precision, batch_size=args.batch_size)
    # check_git_status()
    if device.type == "cpu":
        mixed_precision = False

    try:
        os.makedirs("weights")
    except OSError:
        pass

    # Train
    if not args.evolve:
        tb_writer = SummaryWriter()
        print("Start Tensorboard with `tensorboard --logdir=runs`, view at http://localhost:6006/")
        train(hyper_parameters)

    # Evolve hyper parameters (optional)
    else:
        tb_writer = None
        args.notest, args.nosave = True, True  # only test/save final epoch

        for _ in range(10):  # generations to evolve
            if os.path.exists("evolve.txt"):  # if evolve.txt exists: select best hyps and mutate
                # Select parent(s)
                parent = "single"  # parent selection method: "single" or "weighted"
                x = np.loadtxt("evolve.txt", ndmin=2)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min()  # weights
                if parent == "single" or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == "weighted":
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                mp, s = 0.9, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([1, 1, 1, 1, 1, 1, 1, 0, .1, 1, 0, 1, 1, 1, 1, 1, 1, 1])  # gains
                ng = len(g)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyper_parameters.keys()):  # plt.hist(v.ravel(), 300)
                    hyper_parameters[k] = x[i + 7] * v[i]  # mutate

            # Clip to limits
            keys = ["lr0", "iou_t", "momentum", "weight_decay", "hsv_s", "hsv_v", "translate", "scale", "fl_gamma"]
            limits = [(1e-5, 1e-2), (0.00, 0.70), (0.60, 0.98), (0, 0.001), (0, .9), (0, .9), (0, .9), (0, .9), (0, 3)]
            for k, v in zip(keys, limits):
                hyper_parameters[k] = np.clip(hyper_parameters[k], v[0], v[1])

            # Train mutation
            results = train(hyper_parameters.copy())

            # Write mutation results
            print_mutation(hyper_parameters, results)

            # Plot results
            # plot_evolution_results(hyp)
