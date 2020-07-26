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
import torch.utils.data
import yaml

from yolov4_pytorch.data import LoadImages
from yolov4_pytorch.data import LoadStreams
from yolov4_pytorch.data import check_image_size
from yolov4_pytorch.model import YOLO
from yolov4_pytorch.model import apply_classifier
from yolov4_pytorch.model import load_classifier
from yolov4_pytorch.utils import create_pretrained
from yolov4_pytorch.utils import non_max_suppression
from yolov4_pytorch.utils import plot_one_box
from yolov4_pytorch.utils import scale_coords
from yolov4_pytorch.utils import select_device
from yolov4_pytorch.utils import time_synchronized


def detect(save_image=False):
    # Configure (320, 192) or (416, 256) or (608, 352) for (height, width)
    config_file = args.config_file
    data = args.data

    output = args.output
    source = args.source
    weights = args.weights
    view_image = args.view_image
    save_txt = args.save_txt
    confidence_thresholds = args.confidence_thresholds,
    iou_thresholds = args.iou_thresholds,
    classes = args.classes
    agnostic = args.agnostic_nms
    augment = args.augment

    with open(data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)
    number_classes, names = int(data_dict["number_classes"]), data_dict["names"]

    camera = False
    if source == "0" or source.startswith("rtsp") or source.startswith("http") or source.endswith(".txt"):
        camera = True

    # Initialize
    device = select_device(args.device)
    if os.path.exists(output):
        shutil.rmtree(output)  # delete output folder
    os.makedirs(output)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Create model
    model = YOLO(config_file=config_file, number_classes=number_classes).to(device)
    image_size = check_image_size(args.image_size, stride=model.stride.max())

    # Load model
    model.load_state_dict(torch.load(weights)["state_dict"])
    model.float()
    model.fuse()
    model.eval()
    if half:
        model.half()  # to FP16

        # Second-stage classifier
    classify = False
    if classify:
        # init model
        model_classifier = load_classifier(name="resnet101", number_classes=2)
        # load model
        model_classifier.load_state_dict(torch.load("weights/resnet101.pth", map_location=device))
        model_classifier.to(device)
        model_classifier.eval()
    else:
        model_classifier = None

    # Set Dataloader
    video_path, video_writer = None, None
    if camera:
        view_image = True
        cudnn.benchmark = True
        dataset = LoadStreams(source, image_size=image_size)
    else:
        save_image = True
        dataset = LoadImages(source, image_size=image_size)

    # Get names and colors
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    start_time = time.time()
    # run once
    image = torch.zeros((1, 3, image_size, image_size), device=device)  # init image
    _ = model(image.half() if half else image) if device.type != 'cpu' else None  # run once
    for image_path, image, raw_images, video_capture in dataset:
        image = torch.from_numpy(image).to(device)
        image = image.half() if half else image.float()  # uint8 to fp16/32
        image /= 255.0  # 0 - 255 to 0.0 - 1.0
        if image.ndimension() == 3:
            image = image.unsqueeze(0)

        # Inference
        inference_time = time_synchronized()
        prediction = model(image, augment=augment)[0]
        end_time = time_synchronized()

        # Apply NMS
        prediction = non_max_suppression(prediction=prediction,
                                         confidence_thresholds=confidence_thresholds,
                                         iou_thresholds=iou_thresholds,
                                         classes=classes,
                                         agnostic=agnostic)

        # Apply Classifier
        if classify:
            prediction = apply_classifier(prediction, model_classifier, image, raw_images)

        # Process detections
        for i, detect in enumerate(prediction):  # detections per image
            if camera:  # batch_size >= 1
                p, context, raw_image = image_path[i], f"{i:g}: ", raw_images[i]
            else:
                p, context, raw_image = image_path, "", raw_images

            save_path = str(Path(output) / Path(p).name)
            context += f"{image.shape[2]}*{image.shape[3]} "  # get image size
            if detect is not None and len(detect):
                # Rescale boxes from img_size to im0 size
                detect[:, :4] = scale_coords(image.shape[2:], detect[:, :4], raw_image.shape).round()

                # Print results
                for classes in detect[:, -1].unique():
                    # detections per class
                    number = (detect[:, -1] == classes).sum()
                    context += f"{number} {names[int(classes)]}s, "

                # Write results
                for *xyxy, confidence, classes in detect:
                    if save_txt:  # Write to file
                        with open(save_path + ".txt", "a") as files:
                            files.write(("%e " * 6 + "\n") % (*xyxy, classes, confidence))

                    if save_image or view_image:  # Add bbox to image
                        label = f"{names[int(classes)]} {confidence * 100:.2f}%"
                        plot_one_box(xyxy=xyxy,
                                     image=raw_image,
                                     color=colors[int(classes)],
                                     label=label,
                                     line_thickness=None)

            # Stream results
            if view_image:
                cv2.imshow("camera", raw_image)
                if cv2.waitKey(1) == ord("q"):  # q to quit
                    raise StopIteration

            # Print time (inference + NMS)
            print(f"{context}Done. {end_time - inference_time:.3f}s")

            # Save results (image with detections)
            if save_image:
                if dataset.mode == "images":
                    cv2.imwrite(save_path, raw_image)
                else:
                    if video_path != save_path:  # new video
                        video_path = save_path
                        if isinstance(video_writer, cv2.VideoWriter):
                            video_writer.release()  # release previous video writer

                        fps = video_capture.get(cv2.CAP_PROP_FPS)
                        w = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        video_writer = cv2.VideoWriter(save_path,
                                                       cv2.VideoWriter_fourcc(
                                                           *args.fourcc), fps,
                                                       (w, h))
                    video_writer.write(raw_image)

    print(f"Done. ({time.time() - start_time:.3f}s)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, default="configs/COCO-Detection/yolov5-small.yaml",
                        help="Neural network profile path. (default: `configs/COCO-Detection/yolov5-small.yaml`)")
    parser.add_argument("--data", type=str, default="data/coco2017.yaml",
                        help="Path to dataset. (default: data/coco2017.yaml)")
    parser.add_argument("--image-size", type=int, default=640,
                        help="Size of processing picture. (default: 640)")
    parser.add_argument("--weights", type=str, default="weights/yolov5-small.pth",
                        help="Initial weights path. (default: `weights/yolov5-small.pth`)")
    parser.add_argument("--confidence-thresholds", type=float, default=0.4,
                        help="Object confidence threshold. (default=0.4)")
    parser.add_argument("--iou-thresholds", type=float, default=0.5,
                        help="IOU threshold for NMS. (default=0.5)")
    parser.add_argument("--source", type=str, default="data/examples",
                        help="Image input source. (default: `data/examples`)")
    parser.add_argument("--output", type=str, default="outputs",
                        help="Output result folder. (default: `outputs`)")
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
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument("--device", default="",
                        help="device id i.e. `0` or `0,1` or `cpu`. (default: ``).")
    args = parser.parse_args()
    print(args)

    with torch.no_grad():
        if args.update:  # update all models (to fix SourceChangeWarning)
            for args.weights.split("/")[-1] in ['yolov5-small.pth', 'yolov5-medium.pth', 'yolov5-large.pth',
                                                'yolov5-xlarge.pth', 'yolov3-spp.pth']:
                detect()
                create_pretrained(args.weights, args.weights)
        else:
            detect()
