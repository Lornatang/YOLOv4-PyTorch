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
import yaml

from yolov4_pytorch.data import LoadImages
from yolov4_pytorch.data import LoadStreams
from yolov4_pytorch.data import check_image_size
from yolov4_pytorch.model import YOLO
from yolov4_pytorch.model import apply_classifier
from yolov4_pytorch.model import load_classifier
from yolov4_pytorch.utils import non_max_suppression
from yolov4_pytorch.utils import plot_one_box
from yolov4_pytorch.utils import scale_coords
from yolov4_pytorch.utils import select_device
from yolov4_pytorch.utils import time_synchronized
from yolov4_pytorch.utils import xyxy2xywh


def detect(save_image=False):
    out = args.output
    source = args.source
    weights = args.weights
    half = args.half
    view_image = args.view_image
    save_txt = args.save_txt
    image_size = args.image_size

    webcam = source == "0" or source.startswith("rtsp") or source.startswith("http") or source.endswith(".txt")

    # Initialize
    device = select_device(args.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Initialize model
    model = YOLO(args.config_file).to(device)

    # Load weight
    model.load_state_dict(torch.load(weights, map_location=device)["state_dict"])
    model.float()
    model.eval()

    image_size = check_image_size(image_size, s=model.model[-1].stride.max())  # check image_size

    # Second-stage classifier
    classify = False
    model_classify = None
    if classify:
        model_classify = load_classifier(name="resnet101", classes=2)  # initialize
        model_classify.load_state_dict(torch.load("weights/resnet101.pth", map_location=device))
        model_classify.to(device).eval()

    # Half precision
    half = half and device.type != "cpu"  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    video_path, video_writer = None, None
    if webcam:
        view_image = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, image_size=image_size)
    else:
        save_image = True
        dataset = LoadImages(source, image_size=image_size)

    # Get names and colors
    with open(args.data) as data_file:
        data_dict = yaml.load(data_file, Loader=yaml.FullLoader)  # model dict
    names = data_dict["names"]
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    start_time = time.time()
    image = torch.zeros((1, 3, image_size, image_size), device=device)  # init img
    _ = model(image.half() if half else image.float()) if device.type != "cpu" else None  # run once
    for path, image, raw_images, video_cap in dataset:
        image = torch.from_numpy(image).to(device)
        image = image.half() if half else image.float()  # uint8 to fp16/32
        image /= 255.0  # 0 - 255 to 0.0 - 1.0
        if image.ndimension() == 3:
            image = image.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        predict = model(image, augment=args.augment)[0]

        # to float
        if half:
            predict = predict.float()

        # Apply NMS
        predict = non_max_suppression(predict, args.confidence_threshold, args.iou_threshold,
                                      classes=args.classes, agnostic=args.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            predict = apply_classifier(predict, model_classify, image, raw_images)

        # Process detections
        for i, det in enumerate(predict):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, raw_image = path[i], f"{i}: ", raw_images[i].copy()
            else:
                p, s, raw_image = path, "", raw_images

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ("_%g" % dataset.frame if dataset.mode == "video" else "")
            s += f"{image.shape[2]}x{image.shape[3]} "  # print string
            gn = torch.tensor(raw_image.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to raw_image size
                det[:, :4] = scale_coords(image.shape[2:], det[:, :4], raw_image.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += "%g %ss, " % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh

                        with open(txt_path + ".txt", "a") as f:
                            f.write(("%g " * 5 + "\n") % (cls, *xywh))  # label format

                    if save_image or view_image:  # Add bbox to image
                        label = f"{names[int(cls)]} {int(conf * 100)}%"
                        plot_one_box(xyxy, raw_image, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print(f"{s}Done. ({t2 - t1:.3f}s)")

            # Stream results
            if view_image:
                cv2.imshow(p, raw_image)
                if cv2.waitKey(1) == ord("q"):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_image:
                if dataset.mode == "images":
                    cv2.imwrite(save_path, raw_image)
                else:
                    if video_path != save_path:  # new video
                        video_path = save_path
                        if isinstance(video_writer, cv2.VideoWriter):
                            video_writer.release()  # release previous video writer

                        fps = video_cap.get(cv2.CAP_PROP_FPS)
                        w = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        video_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*args.fourcc), fps, (w, h))
                    video_writer.write(raw_image)

    if save_txt or save_image:
        print(f"Results saved to {os.getcwd() + os.sep + out}.")

    print(f"Done. ({time.time() - start_time:.3f}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, default="configs/COCO-Detection/yolov5-small.yaml",
                        help="Neural network profile path. (default: `configs/COCO-Detection/yolov5-small.yaml`)")
    parser.add_argument("--data", type=str, default="data/coco.yaml",
                        help="Types of objects detected. (default: data/coco.yaml)")
    parser.add_argument("--weights", type=str, default="weights/yolov5-small.pth",
                        help="Model file weight path. (default: `weights/yolov5-small.pth`)")
    parser.add_argument("--source", type=str, default="data/examples",
                        help="Image input source. (default: `data/examples`)")
    parser.add_argument("--output", type=str, default="outputs",
                        help="Output result folder. (default: `outputs`)")
    parser.add_argument("--image-size", type=int, default=640,
                        help="Size of processing picture. (default: 640)")
    parser.add_argument("--confidence-threshold", type=float, default=0.4,
                        help="Object confidence threshold. (default=0.4)")
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                        help="IOU threshold for NMS. (default=0.5)")
    parser.add_argument("--fourcc", type=str, default="mp4v",
                        help="output video codec (verify ffmpeg support). (default=mp4v)")
    parser.add_argument("--half", action="store_true", help="half precision FP16 inference")
    parser.add_argument("--device", default="0",
                        help="device id (i.e. 0 or 0,1) or cpu. (default: `0`).")
    parser.add_argument("--view-image", action="store_true",
                        help="Display results")
    parser.add_argument("--save-txt", action="store_true",
                        help="Save results to *.txt")
    parser.add_argument("--classes", nargs="+", type=int,
                        help="Filter by class")
    parser.add_argument("--agnostic-nms", action="store_true",
                        help="Class-agnostic NMS")
    parser.add_argument("--augment", action="store_true",
                        help="augmented inference")
    args = parser.parse_args()
    args.image_size = check_image_size(args.image_size)
    print(args)

    with torch.no_grad():
        detect()
