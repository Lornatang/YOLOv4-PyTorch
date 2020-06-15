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
import os
import random
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

from yolov4_pytorch import LoadImages
from yolov4_pytorch import LoadStreams
from yolov4_pytorch import apply_classifier
from yolov4_pytorch import check_img_size
from yolov4_pytorch import load_classifier
from yolov4_pytorch import non_max_suppression
from yolov4_pytorch import plot_one_box
from yolov4_pytorch import scale_coords
from yolov4_pytorch import select_device
from yolov4_pytorch import time_synchronized
from yolov4_pytorch import xyxy2xywh


def detect(save_img=False):
    out, source, weights, half, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.half, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == "0" or source.startswith("rtsp") or source.startswith("http") or source.endswith(".txt")

    # Initialize
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Load model
    model = torch.load(weights, map_location=device)["model"]
    # torch.save(torch.load(weights, map_location=device), weights)  # update model if SourceChangeWarning
    # model.fuse()
    model.to(device).eval()

    # Second-stage classifier
    classify = False
    modelc = None
    if classify:
        modelc = load_classifier(name="resnet101", classes=2)  # initialize
        modelc.load_state_dict(torch.load("weights/resnet101.pth", map_location=device)["model"])  # load weights
        modelc.to(device).eval()

    # Half precision
    half = half and device.type != "cpu"  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, image_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, image_size=imgsz)

    # Get names and colors
    names = model.names if hasattr(model, "names") else model.modules.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img.float()) if device.type != "cpu" else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # to float
        if half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                   fast=True, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], "%g: " % i, im0s[i].copy()
            else:
                p, s, im0 = path, "", im0s

            save_path = str(Path(out) / Path(p).name)
            s += "%gx%g " % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += "%g %ss, " % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(save_path[:save_path.rfind(".")] + ".txt", "a") as file:
                            file.write(("%g " * 5 + "\n") % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = "%s %.2f" % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print("%sDone. (%.3fs)" % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord("q"):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == "images":
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print("Results saved to %s" % os.getcwd() + os.sep + out)

    print("Done. (%.3fs)" % (time.time() - t0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="weights/yolov5s.pth", help="model.pt path")
    parser.add_argument("--source", type=str, default="inference/images", help="source")  # file/folder, 0 for webcam
    parser.add_argument("--output", type=str, default="inference/output", help="output folder")  # output folder
    parser.add_argument("--img-size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument("--conf-thres", type=float, default=0.4, help="object confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.5, help="IOU threshold for NMS")
    parser.add_argument("--fourcc", type=str, default="mp4v", help="output video codec (verify ffmpeg support)")
    parser.add_argument("--half", action="store_true", help="half precision FP16 inference")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="display results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    opt = parser.parse_args()
    opt.img_size = check_img_size(opt.img_size)
    print(opt)

    with torch.no_grad():
        detect()
