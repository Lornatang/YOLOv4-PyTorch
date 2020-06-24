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
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from yolov4_pytorch.data import LoadImagesAndLabels
from yolov4_pytorch.data import check_image_size
from yolov4_pytorch.model import YOLO
from yolov4_pytorch.model import model_info
from yolov4_pytorch.utils import ap_per_class
from yolov4_pytorch.utils import box_iou
from yolov4_pytorch.utils import clip_coords
from yolov4_pytorch.utils import coco80_to_coco91_class
from yolov4_pytorch.utils import compute_loss
from yolov4_pytorch.utils import non_max_suppression
from yolov4_pytorch.utils import output_to_target
from yolov4_pytorch.utils import plot_images
from yolov4_pytorch.utils import scale_coords
from yolov4_pytorch.utils import select_device
from yolov4_pytorch.utils import time_synchronized
from yolov4_pytorch.utils import xywh2xyxy
from yolov4_pytorch.utils import xyxy2xywh


def evaluate(config_file,
             data,
             weights=None,
             batch_size=16,
             image_size=640,
             confidence_threshold=0.001,
             iou_threshold=0.6,  # for nms
             save_json=False,
             single_cls=False,
             augment=False,
             verbose=False,
             model=None,
             dataloader=None,
             merge=False):
    # Initialize/load model and set device
    if model is None:
        training = False
        device = select_device(args.device, batch_size=batch_size)
        half = device.type != "cpu"  # half precision only supported on CUDA

        # Remove previous
        for filename in glob.glob("test_batch_*.png"):
            os.remove(filename)

        # Configure
        with open(args.data) as data_file:
            data_dict = yaml.load(data_file, Loader=yaml.FullLoader)  # model dict
        num_classes = 1 if args.single_cls else int(data_dict["num_classes"])  # number of classes

        # Create model
        model = YOLO(config_file).to(device)
        assert model.config_file[
                   "num_classes"] == num_classes, f"{args.data} num classes={num_classes} classes but {config_file} classes={config_file['num_classes']} classes "

        # Load model
        model.load_state_dict(torch.load(weights, map_location=device)["state_dict"])
        model.float()
        model.fuse()
        model.to(device)

        model_info(model)

        if half:
            model.half()

    else:  # called by train.py
        training = True
        device = next(model.parameters()).device  # get model device

        half = False
        if half:
            model.half()

    # Configure run
    model.eval()
    with open(data) as filename:
        data = yaml.load(filename, Loader=yaml.FullLoader)  # model dict
    num_classes = 1 if single_cls else int(data["num_classes"])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if dataloader is None:
        image = torch.zeros((1, 3, image_size, image_size), device=device)  # init img
        _ = model(image.half() if half else image.float()) if device.type != "cpu" else None  # run once

        merge = args.merge  # use Merge NMS
        path = data["test"] if args.task == "test" else data["val"]  # path to val/test images
        dataset = LoadImagesAndLabels(path,
                                      image_size,
                                      batch_size,
                                      rect=True,  # rectangular inference
                                      single_cls=args.single_cls,  # single class mode
                                      pad=0.5)  # padding
        batch_size = min(batch_size, len(dataset))
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=args.workers,
                                pin_memory=True,
                                collate_fn=dataset.collate_fn)

    seen = 0
    names = data["names"]
    coco91class = coco80_to_coco91_class()
    s = ("%20s" + "%12s" * 6) % ("Class", "Images", "Targets", "P", "R", "mAP@.5", "mAP@.5:.95")
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    for batch_i, (images, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        images = images.to(device)
        images = images.half() if half else images.float()  # uint8 to fp16/32
        images /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        batch_size, _, height, width = images.shape  # batch size, channels, height, width
        whwh = torch.Tensor([width, height, width, height]).to(device)

        # Disable gradients
        with torch.no_grad():
            # Run model
            t = time_synchronized()
            inf_out, train_out = model(images, augment=augment)  # inference and training outputs
            t0 += time_synchronized() - t

            # Compute loss
            if training:  # if model has loss hyper parameters
                loss += compute_loss([x.float() for x in train_out], targets, model)[1][:3]  # GIoU, obj, cls

            # Run NMS
            t = time_synchronized()
            output = non_max_suppression(inf_out,
                                         confidence_threshold=confidence_threshold,
                                         iou_threshold=iou_threshold,
                                         merge=merge)
            t1 += time_synchronized() - t

        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if pred is None:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Clip boxes to image bounds
            clip_coords(pred, (height, width))

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(Path(paths[si]).stem.split("_")[-1])
                box = pred[:, :4].clone()  # xyxy
                scale_coords(images[si].shape[1:], box, shapes[si][0], shapes[si][1])  # to original shape
                box = xyxy2xywh(box)  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({"image_id": image_id,
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
                    ti = (cls == tcls_tensor).nonzero().view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero().view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        for j in (ious > iouv[0]).nonzero():
                            d = ti[i[j]]  # detected target
                            if d not in detected:
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        # Plot images
        if batch_i < 1:
            filename = f"test_batch_{batch_i}_gt.png"  # filename
            plot_images(images, targets, paths, filename, names)  # ground truth
            filename = f"test_batch_{batch_i}_pred.png"
            plot_images(images, output_to_target(output, width, height), paths, filename, names)  # predictions

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats):
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=num_classes)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = "%20s" + "%12.3g" * 6  # print format
    print(pf % ("all", seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if verbose and num_classes > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (image_size, image_size, batch_size)  # tuple
    if not training:
        print("Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g" % t)

    # Save JSON
    if save_json and map50 and len(jdict):
        imgIds = [int(Path(x).stem.split("_")[-1]) for x in dataloader.dataset.image_files]
        filename = f"detections_val2017_{(weights.split(os.sep)[-1].replace('.pth', '') if weights else '')}results.json"
        print("\nCOCO mAP with pycocotools... saving %s..." % filename)
        with open(filename, "w") as file:
            json.dump(jdict, file)

        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            cocoGt = COCO(
                glob.glob("data/COCO-Detection/annotations/instances_val*.json")[0])  # initialize COCO ground truth api
            cocoDt = cocoGt.loadRes(filename)  # initialize COCO pred api

            cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
            cocoEval.params.imgIds = imgIds  # [:32]  # only evaluate these images
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            map, map50 = cocoEval.stats[:2]  # update to pycocotools results (mAP@0.5:0.95, mAP@0.5)
        except:
            print("WARNING: pycocotools must be installed with numpy==1.17 to run correctly. "
                  "See https://github.com/cocodataset/cocoapi/issues/356")

    # Return results
    maps = np.zeros(num_classes) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, default="configs/COCO-Detection/yolov5-small.yaml",
                        help="Neural network profile path. (default: `configs/COCO-Detection/yolov5-small.yaml`)")
    parser.add_argument("--data", type=str, default="data/coco2014.yaml",
                        help="Path to dataset. (default: data/coco2014.yaml)")
    parser.add_argument("--weights", type=str, default="weights/yolov5-small.pth",
                        help="Initial weights path. (default: `weights/yolov5-small.pth`)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Size of each image batch. (default=32)")
    parser.add_argument("--workers", default=4, type=int, metavar="N",
                        help="Number of data loading workers (default: 4)")
    parser.add_argument("--image-size", type=int, default=640,
                        help="Size of processing picture. (default=640)")
    parser.add_argument("--confidence-threshold", type=float, default=0.001,
                        help="Object confidence threshold. (default=0.001)")
    parser.add_argument("--iou-threshold", type=float, default=0.65,
                        help="IOU threshold for NMS. (default=0.65)")
    parser.add_argument("--task", default="eval", help="`eval`, `study`, `test`")
    parser.add_argument("--device", default="", help="device id (i.e. 0 or 0,1) or cpu")
    parser.add_argument("--save-json", action="store_true",
                        help="save a cocoapi-compatible JSON results file")
    parser.add_argument("--single-cls", action="store_true", help="train as single-class dataset")
    parser.add_argument("--augment", action="store_true", help="augmented for testing")
    parser.add_argument("--merge", action="store_true", help="use Merge NMS")
    parser.add_argument("--verbose", action="store_true", help="report mAP by class")

    args = parser.parse_args()
    args.image_size = check_image_size(args.image_size)
    args.save_json = args.save_json or args.data.endswith("coco2014.yaml") or args.data.endswith("coco2017.yaml")
    print(args)

    # task = "val", "test", "study"
    if args.task in ["eval", "test"]:  # (default) run normally
        evaluate(args.config_file,
                 args.data,
                 args.weights,
                 args.batch_size,
                 args.image_size,
                 args.confidence_threshold,
                 args.iou_threshold,
                 args.save_json,
                 args.single_cls,
                 args.augment,
                 args.verbose)

    elif args.task == "study":  # run over a range of settings and save/plot
        for weights in ["yolov3-spp.pth", "yolov5-small.pth", "yolov5-medium.pth", "yolov5-large.pth",
                        "yolov5-xlarge.pth"]:
            f = "study_%s_%s.txt" % (Path(args.data).stem, Path(weights).stem)  # filename to save to
            x = list(range(352, 832, 64))  # x axis
            y = []  # y axis
            for i in x:  # image-size
                print("\nRunning %s point %s..." % (f, i))
                r, _, t = evaluate(args.data, weights, args.batch_size, i, args.confidence_threshold,
                                   args.iou_threshold, args.save_json)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt="%10.4g")  # save
        os.system("zip -r study.zip study_*.txt")
