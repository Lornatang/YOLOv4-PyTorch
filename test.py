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
import json
import os
import shutil

import numpy as np
import torch
import torch.distributed
import torch.utils.data
import yaml
from tqdm import tqdm

from yolov4_pytorch.data import create_dataloader
from yolov4_pytorch.model import YOLO
from yolov4_pytorch.utils import ap_per_class
from yolov4_pytorch.utils import box_iou
from yolov4_pytorch.utils import clip_coords
from yolov4_pytorch.utils import coco80_to_coco91_class
from yolov4_pytorch.utils import compute_loss
from yolov4_pytorch.utils import non_max_suppression
from yolov4_pytorch.utils import scale_coords
from yolov4_pytorch.utils import select_device
from yolov4_pytorch.utils import time_synchronized
from yolov4_pytorch.utils import xywh2xyxy
from yolov4_pytorch.utils import xyxy2xywh


def evaluate(config_file="configs/COCO-Detection/yolov5-small.yaml",
             batch_size=16,
             data="data/coco2017.yaml",
             image_size=640,
             weights=None,
             confidence_thresholds=0.001,
             iou_thresholds=0.6,
             save_json=False,
             merge=False,
             augment=False,
             verbose=False,
             save_txt=False,
             model=None,
             dataloader=None):
    with open(data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)
    number_classes, names = int(data_dict["number_classes"]), data_dict["names"]
    assert len(names) == number_classes, f"{len(names)} names found for nc={number_classes} dataset in {data}"

    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        device = select_device(args.device, batch_size=args.batch_size)
        if save_txt:
            if os.path.exists("outputs"):
                shutil.rmtree("outputs")  # delete output folder
            os.makedirs("outputs")  # make new output folder

        # Create model
        model = YOLO(config_file=config_file, number_classes=number_classes).to(device)

        # Load model
        model.load_state_dict(torch.load(weights)["state_dict"])
        model.float()
        model.fuse()
        model.eval()

    # Half
    half = device.type != "cpu"  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()

    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if not training:
        image = torch.zeros((1, 3, image_size, image_size), device=device)  # init image
        _ = model(image.half() if half else image) if device.type != "cpu" else None  # run once
        dataroot = data_dict["test"] if data_dict["test"] else data_dict["val"]  # path to val/test images

        dataset, dataloader = create_dataloader(dataroot=dataroot,
                                                image_size=image_size,
                                                batch_size=batch_size,
                                                hyper_parameters=None,
                                                augment=False,
                                                cache=False,
                                                rect=True)

    seen = 0
    coco91class = coco80_to_coco91_class()
    context = f"{'Class':>20}{'Images':>12}{'Targets':>12}{'P':>12}{'R':>12}{'mAP@.5':>12}{'mAP@.5:.95':>12}"
    p, r, f1, mp, mr, map50, map, inference_time, nms_time = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    for _, (image, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=context)):
        image = image.to(device, non_blocking=True)
        image = image.half() if half else image.float()  # uint8 to fp16/32
        image /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = image.shape  # batch size, channels, height, width
        whwh = torch.Tensor([width, height, width, height]).to(device)

        # Disable gradients
        with torch.no_grad():
            # Run model
            t = time_synchronized()
            prediction, outputs = model(image, augment=augment)  # inference and training outputs
            inference_time += time_synchronized() - t

            # Compute loss
            if training:  # if model has loss hyper parameters
                loss += compute_loss([x.float() for x in outputs], targets, model)[1][:3]  # GIoU, obj, cls

            # Run NMS
            t = time_synchronized()
            prediction = non_max_suppression(prediction=prediction,
                                             confidence_thresholds=confidence_thresholds,
                                             iou_thresholds=iou_thresholds,
                                             merge=merge,
                                             classes=None,
                                             agnostic=False)
            nms_time += time_synchronized() - t

        # Statistics per image
        for si, pred in enumerate(prediction):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if pred is None:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Append to text file
            if save_txt:
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                txt_path = os.path.join("outputs", paths[si].split("/")[-1][:-4])
                pred[:, :4] = scale_coords(image[si].shape[1:], pred[:, :4], shapes[si][0],
                                           shapes[si][1])  # to original
                for *xyxy, conf, cls in pred:
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    with open(txt_path + ".txt", "a") as f:
                        f.write(("%g " * 5 + "\n") % (cls, *xywh))  # label format

            # Clip boxes to image bounds
            clip_coords(pred, (height, width))

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = si[:-4]
                box = pred[:, :4].clone()  # xyxy
                scale_coords(image[si].shape[1:], box, shapes[si][0], shapes[si][1])  # to original shape
                box = xyxy2xywh(box)  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({"image_id": int(image_id) if image_id.isnumeric() else image_id,
                                  "category_id": coco91class[int(p[5])],
                                  "bbox": [round(x, 3) for x in b],
                                  "score": round(p[4], 5)})

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5]) * whwh

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d not in detected:
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=number_classes)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    print(f"{'all':>20}{seen:>12}{nt.sum():>12}{mp:>12.3f}{mr:>12.3f}{map50:>12.3f}{map:>12.3f}")

    # Print results per class

    if verbose:
        for i, c in enumerate(ap_class):
            print(f"{names[c]:>20}{seen:>12}{nt[c]:>12}{p[i]:>12.3f}{r[i]:>12.3f}{ap50[i]:>12.3f}{ap[i]:>12.3f}")

    # Print speeds
    if not training:
        print("Speed: "
              f"{inference_time / seen * 1000:.1f}/"
              f"{nms_time / seen * 1000:.1f}/"
              f"{(inference_time + nms_time) / seen * 1000:.1f} ms "
              f"inference/NMS/total per {image_size}x{image_size} image at batch-size {batch_size}")

    # Save JSON
    if save_json and len(jdict):
        f = f"detections_val2017_{weights.split('/')[-1].replace('.pth', '')}_results.json"
        print(f"\nCOCO mAP with pycocotools... saving {f}...")
        with open(f, "w") as file:
            json.dump(jdict, file)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            imgIds = [int(x.split("/")[-1][:-4]) for x in dataloader.dataset.image_files]
            cocoGt = COCO(glob.glob("data/coco2017/annotations/instances_val*.json")[0])
            cocoDt = cocoGt.loadRes(f)  # initialize COCO pred api
            cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
            cocoEval.params.imgIds = imgIds  # image IDs to evaluate
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            map, map50 = cocoEval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            print(f"ERROR: pycocotools unable to run: {e}")

    # Return results
    model.float()  # for training
    maps = np.zeros(int(data_dict["number_classes"])) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(usage="\n\tpython test.py --config-file configs/COCO-Detection/yolov5-small.yaml"
                                           " --data data/coco2017.yaml"
                                           " --weights weights/COCO-Detection/yolov5-small.pth")
    parser.add_argument("--config-file", type=str, default="configs/COCO-Detection/yolov5-small.yaml",
                        help="Neural network profile path. (default: `configs/COCO-Detection/yolov5-small.yaml`)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="mini-batch size (default: 32), this is the total "
                             "batch size of all GPUs on the current node when "
                             "using Data Parallel or Distributed Data Parallel")
    parser.add_argument("--data", type=str, default="data/coco2017.yaml",
                        help="Path to dataset. (default: data/coco2017.yaml)")
    parser.add_argument("--image-size", type=int, default=640,
                        help="Size of processing picture. (default: 640)")
    parser.add_argument("--weights", type=str, default="weights/COCO-Detection/yolov5-small.pth",
                        help="Initial weights path. (default: `weights/COCO-Detection/yolov5-small.pth`)")
    parser.add_argument("--confidence-thresholds", type=float, default=0.001,
                        help="Object confidence threshold. (default=0.001)")
    parser.add_argument("--iou-thresholds", type=float, default=0.65,
                        help="IOU threshold for NMS. (default=0.65)")
    parser.add_argument("--save-json", action="store_true",
                        help="save a cocoapi-compatible JSON results file")
    parser.add_argument("--augment", action="store_true",
                        help="augmented inference")
    parser.add_argument("--merge", action="store_true", help="use Merge NMS")
    parser.add_argument("--verbose", action="store_true", help="report mAP by class")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--device", default="",
                        help="device id i.e. `0` or `0,1` or `cpu`. (default: ``).")
    args = parser.parse_args()
    args.save_json |= args.data.endswith("coco2014.yaml") or args.data.endswith("coco2017.yaml")

    print(args)

    evaluate(config_file=args.config_file,
             batch_size=args.batch_size,
             data=args.data,
             image_size=args.image_size,
             weights=args.weights,
             confidence_thresholds=args.confidence_thresholds,
             iou_thresholds=args.iou_thresholds,
             save_json=args.save_json,
             merge=args.merge,
             augment=args.augment,
             verbose=args.verbose,
             save_txt=args.save_txt)
