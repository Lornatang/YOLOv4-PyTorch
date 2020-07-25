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
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from apex import amp
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import test
from yolov4_pytorch.data import check_image_size
from yolov4_pytorch.data import create_dataloader
from yolov4_pytorch.model import YOLO
from yolov4_pytorch.solver import ModelEMA
from yolov4_pytorch.utils import attempt_download
from yolov4_pytorch.utils import check_anchors
from yolov4_pytorch.utils import check_file
from yolov4_pytorch.utils import compute_loss
from yolov4_pytorch.utils import fitness
from yolov4_pytorch.utils import get_latest_run
from yolov4_pytorch.utils import increment_dir
from yolov4_pytorch.utils import init_seeds
from yolov4_pytorch.utils import labels_to_class_weights
from yolov4_pytorch.utils import labels_to_image_weights
from yolov4_pytorch.utils import select_device
from yolov4_pytorch.utils import torch_distributed_zero_first

# Hyper parameters
hyper_parameters = {'lr0': 0.01,  # initial learning rate (SGD=1E-2, Adam=1E-3)
                    'momentum': 0.937,  # SGD momentum/Adam beta1
                    'weight_decay': 5e-4,  # optimizer weight decay
                    'giou': 0.05,  # giou loss gain
                    'cls': 0.5,  # cls loss gain
                    'cls_pw': 1.0,  # cls BCELoss positive_weight
                    'obj': 1.0,  # obj loss gain (*=img_size/320 if img_size != 320)
                    'obj_pw': 1.0,  # obj BCELoss positive_weight
                    'iou_t': 0.20,  # iou training threshold
                    'anchor_t': 4.0,  # anchor-multiple threshold
                    'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
                    'hsv_h': 0.015,  # image HSV-Hue augmentation (fraction)
                    'hsv_s': 0.7,  # image HSV-Saturation augmentation (fraction)
                    'hsv_v': 0.4,  # image HSV-Value augmentation (fraction)
                    'degrees': 0.0,  # image rotation (+/- deg)
                    'translate': 0.0,  # image translation (+/- fraction)
                    'scale': 0.5,  # image scale (+/- gain)
                    'shear': 0.0}  # image shear (+/- deg)


def train(parameters, tb_writer, opt, device):
    print(f'Hyper parameters {parameters}')
    epochs = opt.epochs
    batch_size = opt.batch_size
    total_batch_size = opt.batch_size
    weights = opt.weights
    rank = opt.local_rank

    # Configure
    init_seeds(2 + rank)

    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)

    train_path = data_dict['train']
    test_path = data_dict['val']

    number_classes, names = (1, ['item']) if opt.single_cls else (int(data_dict['nc']), data_dict['names'])
    assert len(names) == number_classes, f"{len(names)} names found for nc={number_classes} dataset in {opt.data}"

    # Remove previous results
    if os.path.exists("results.txt"):
        os.remove("results.txt")

    # Create model
    model = YOLO(opt.cfg, nc=number_classes).to(device)

    # Image sizes
    gs = int(max(model.stride))  # grid size (max stride)
    image_size = check_image_size(opt.img_size, gs)

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    parameters['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_parameters():
        if v.requires_grad:
            if '.bias' in k:
                pg2.append(v)  # biases
            elif '.weight' in k and '.bn' not in k:
                pg1.append(v)  # apply weight decay
            else:
                pg0.append(v)  # all else

    optimizer = torch.optim.SGD(pg0, lr=parameters['lr0'], momentum=parameters['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': parameters['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    print(f"Optimizer groups: {len(pg2)} .bias, {len(pg1)} conv.weight, {len(pg0)} other")
    del pg0, pg1, pg2

    # Load Model
    with torch_distributed_zero_first(rank):
        attempt_download(weights)
    start_epoch, best_fitness = 0, 0.0

    if os.path.exists(weights):
        checkpoint = torch.load(weights, map_location=device)  # load checkpoint

        # load model
        try:
            exclude = ['anchor']  # exclude keys
            checkpoint['state_dict'] = {k: v for k, v in checkpoint['state_dict'].float().state_dict().items()
                                        if k in model.state_dict() and not any(x in k for x in exclude)
                                        and model.state_dict()[k].shape == v.shape}
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        except KeyError as e:
            s = "The model parameter file does not match!"
            raise KeyError(s) from e

        # load optimizer
        if checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_fitness = checkpoint['best_fitness']

        # load results
        if checkpoint.get('training_results') is not None:
            with open("results.txt", 'w') as file:
                file.write(checkpoint['training_results'])  # write results.txt

        # load epochs
        start_epoch = checkpoint['epoch'] + 1
        if epochs < start_epoch:
            print(f"{weights} has been trained for {checkpoint['epoch']} epochs. "
                  f"Fine-tuning for {epochs} additional epochs.")
            epochs += checkpoint['epoch']  # fine tune additional epochs

        del checkpoint

    # Mixed precision training https://github.com/NVIDIA/apex
    model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.8 + 0.2  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # DP mode
    if device.type != 'cpu' and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt.sync_bn and device.type != 'cpu' and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        print('Using SyncBatchNorm()')

    # Exponential moving average
    ema = ModelEMA(model) if rank in [-1, 0] else None

    # DDP mode
    if device.type != 'cpu' and rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank)

    # Train dataloader
    train_dataloader, train_dataset = create_dataloader(train_path, image_size, batch_size, gs, opt,
                                                        hyp=parameters,
                                                        augment=True,
                                                        cache=opt.cache_images,
                                                        rect=opt.rect,
                                                        local_rank=rank,
                                                        world_size=opt.world_size)
    mlc = np.concatenate(train_dataset.labels, 0)[:, 0].max()  # max label class
    nb = len(train_dataloader)  # number of batches
    assert mlc < number_classes, f"Label class {mlc} exceeds nc={number_classes} in {opt.data}. " \
                                 f"Possible class labels are 0-{number_classes - 1}"

    # Test dataloader
    test_dataloader, _ = create_dataloader(test_path, image_size, total_batch_size, gs, opt,
                                           hyp=parameters,
                                           augment=False,
                                           cache=opt.cache_images,
                                           rect=True,
                                           local_rank=rank,
                                           world_size=opt.world_size)

    # Model parameters
    parameters['cls'] *= number_classes / 80.  # scale coco-tuned hyp['cls'] to current dataset
    model.nc = number_classes  # attach number of classes to model
    model.hyp = parameters  # attach hyperparameters to model
    model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
    model.class_weights = labels_to_class_weights(train_dataset.labels, number_classes).to(device)

    # Class frequency
    labels = np.concatenate(train_dataset.labels, 0)
    c = torch.tensor(labels[:, 0])  # classes
    if tb_writer:
        tb_writer.add_histogram('classes', c, 0)

    # Check anchors
    if not opt.noautoanchor:
        check_anchors(train_dataset, model=model, thr=parameters['anchor_t'], imgsz=image_size)

    # Start training
    t0 = time.time()
    nw = max(3 * nb, 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    maps = np.zeros(number_classes)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    scheduler.last_epoch = start_epoch - 1  # do not move

    print(f"Image sizes {image_size} train, {image_size} test")
    print(f"Using {train_dataloader.num_workers} dataloader workers")
    print(f"Starting training for {epochs} epochs...")

    for epoch in range(start_epoch, epochs):
        model.train()

        # Update image weights (optional)
        # When in DDP mode, the generated indices will be broadcasted to synchronize dataset.
        if train_dataset.image_weights:
            # Generate indices.
            if rank in [-1, 0]:
                w = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
                image_weights = labels_to_image_weights(train_dataset.labels, nc=number_classes, class_weights=w)
                train_dataset.indices = random.choices(range(train_dataset.n), weights=image_weights,
                                                       k=train_dataset.n)  # rand weighted idx
            # Broadcast.
            if rank != -1:
                indices = torch.zeros([train_dataset.n], dtype=torch.int)
                if rank == 0:
                    indices[:] = torch.from_tensor(train_dataset.indices, dtype=torch.int)
                dist.broadcast(indices, 0)
                if rank != 0:
                    train_dataset.indices = indices.cpu().numpy()

        mloss = torch.zeros(4, device=device)  # mean losses
        if rank != -1:
            train_dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(train_dataloader)
        if rank in [-1, 0]:
            print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))
            pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [0.9, parameters['momentum']])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(image_size * 0.5, image_size * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            pred = model(imgs)

            # Loss
            loss, loss_items = compute_loss(pred, targets.to(device), model)  # scaled by batch_size
            if rank != -1:
                loss *= opt.world_size  # gradient averaged between devices in DDP mode
            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss_items)
                return results

            # Backward
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            # Optimize
            if ni % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()
                if ema is not None:
                    ema.update(model)

            # Print
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.4g' * 6) % (
                    '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
                pbar.set_description(s)

            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        scheduler.step()

        # Only the first process in DDP mode is allowed to log or save checkpoints.
        if rank in [-1, 0]:
            # mAP
            if ema is not None:
                ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'stride'])
            final_epoch = epoch + 1 == epochs
            if not opt.notest or final_epoch:  # Calculate mAP
                results, maps, times = test.test(opt.data,
                                                 batch_size=total_batch_size,
                                                 imgsz=image_size,
                                                 save_json=final_epoch and opt.data.endswith(os.sep + 'coco.yaml'),
                                                 model=ema.ema.module if hasattr(ema.ema, 'module') else ema.ema,
                                                 single_cls=opt.single_cls,
                                                 dataloader=test_dataloader)

                # Write
                with open("results.txt", 'a') as f:
                    f.write(s + '%10.4g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)

                # Tensorboard
                if tb_writer:
                    tags = ['train/giou_loss', 'train/obj_loss', 'train/cls_loss',
                            'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                            'val/giou_loss', 'val/obj_loss', 'val/cls_loss']
                    for x, tag in zip(list(mloss[:-1]) + list(results), tags):
                        tb_writer.add_scalar(tag, x, epoch)

                # Update best mAP
                fi = fitness(np.array(results).reshape(1, -1))  # fitness_i = weighted combination of [P, R, mAP, F1]
                if fi > best_fitness:
                    best_fitness = fi

                # Save model
                with open("results.txt", 'r') as f:  # create checkpoint
                    checkpoint = {'epoch': epoch,
                                  'best_fitness': best_fitness,
                                  'training_results': f.read(),
                                  'model': ema.ema.module.state_dict() if hasattr(ema,
                                                                                  'module') else ema.ema.state_dict(),
                                  'optimizer': None if final_epoch else optimizer.state_dict()}

                # Save last, best and delete
                torch.save(checkpoint, "weights/checkpoint.pth")
                if (best_fitness == fi) and not final_epoch:
                    torch.save(checkpoint, "weights/model_best.pth")
                del checkpoint

        # Finish
        print('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))

    dist.destroy_process_group() if rank not in [-1, 0] else None
    torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='models/yolov5-small.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='data.yaml path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help="Total batch size for all gpus.")
    parser.add_argument('--img-size', type=int, default=640, help='train,test sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', type=str, action='store_true',
                        help='resume from weights/checkpoint.pth, or most recent run if blank.')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    opt = parser.parse_args()

    opt.weights = "weights/checkpoint" if opt.resume and not opt.weights else opt.weights

    opt.cfg = check_file(opt.cfg)  # check file
    opt.data = check_file(opt.data)  # check file

    device = select_device(opt.device, batch_size=opt.batch_size)
    opt.total_batch_size = opt.batch_size
    opt.world_size = 1
    if device.type == 'cpu':
        mixed_precision = False
    elif opt.local_rank != -1:
        # DDP mode
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device("cuda", opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend

        opt.world_size = dist.get_world_size()
        assert opt.batch_size % opt.world_size == 0, "Batch size is not a multiple of the number of devices given!"
        opt.batch_size = opt.total_batch_size // opt.world_size
    print(opt)

    # Train
    if opt.local_rank in [-1, 0]:
        print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
        tb_writer = SummaryWriter(log_dir=increment_dir('runs/exp', opt.name))
    else:
        tb_writer = None
    train(hyper_parameters, tb_writer, opt, device)
