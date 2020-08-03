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
import os
import random
import urllib.request

import cv2
import torch
import yaml
from django.shortcuts import render
from rest_framework.views import APIView

from yolov4_pytorch.data import LoadImages
from yolov4_pytorch.model import YOLO
from yolov4_pytorch.utils import non_max_suppression
from yolov4_pytorch.utils import plot_one_box
from yolov4_pytorch.utils import scale_coords
from yolov4_pytorch.utils import select_device

device = select_device()

# move the model to GPU for speed if available
model = YOLO("../configs/COCO-Detection/mobilenet-v1.yaml").to(device)
# Load weight
model.load_state_dict(
    torch.load("../weights/COCO-Detection/mobilenetv1.pth", map_location=device)["state_dict"])
model.float()
model.fuse()
model.eval()

# Half precision
half = device.type != "cpu"  # half precision only supported on CUDA
if half:
    model.half()

# Get names and colors
with open("../data/coco2017.yaml") as data_file:
    data_dict = yaml.load(data_file, Loader=yaml.FullLoader)  # model dict
names = data_dict["names"]
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]


def preprocess(filename):
    # Set Dataloader
    dataset = LoadImages(filename)
    return dataset


def index(request):
    r""" Get the image based on the base64 encoding or url address
        and do the pencil style conversion
    Args:
      request: Post request in url.
        - image_code: 64-bit encoding of images.
        - url:        The URL of the image.
    Return:
      Base64 bit encoding of the image.
    Notes:
      Later versions will not contexturn an image's address,
      but instead a base64-bit encoded address
    """

    return render(request, "index.html")


class YOLOv4(APIView):

    @staticmethod
    def get(request):
        """ Get the image based on the base64 encoding or url address
        Args:
          request: Post request in url.
            - image_code: 64-bit encoding of images.
            - url:        The URL of the image.
        Return:
          Base64 bit encoding of the image.
        Notes:
          Later versions will not contexturn an image's address,
          but instead a base64-bit encoded address
        """

        base_path = "static/images"

        try:
            os.makedirs(base_path)
        except OSError:
            pass

        filename = os.path.join(base_path, "raw.png")
        if os.path.exists(filename):
            os.remove(filename)

        context = {
            "status_code": 20000
        }
        return render(request, "image.html", context)

    @staticmethod
    def post(request):
        """ Get the image based on the base64 encoding or url address
        Args:
            request: Post request in url.
            - image_code: 64-bit encoding of images.
            - url:        The URL of the image.
        Return:
            Base64 bit encoding of the image.
        Notes:
            Later versions will not contexturn an image's address,
            but instead a base64-bit encoded address
        """
        # Get the url for the image
        url = request.POST.get("url")
        base_path = "static/images"

        try:
            os.makedirs(base_path)
        except OSError:
            pass

        filename = os.path.join(base_path, "raw.png")

        image = urllib.request.urlopen(url)
        with open(filename, "wb") as v:
            v.write(image.read())

        dataset = preprocess(filename)

        msg = None

        for filename, image, raw_images, video_cap in dataset:
            image = torch.from_numpy(image).to(device)
            image = image.half() if half else image.float()  # uint8 to fp16/32
            image /= 255.0  # 0 - 255 to 0.0 - 1.0
            if image.ndimension() == 3:
                image = image.unsqueeze(0)

            # Inference
            prediction = model(image, augment=False)[0]

            # Apply NMS
            prediction = non_max_suppression(prediction, 0.4, 0.5)

            # Process detections
            for i, detect in enumerate(prediction):  # detections per image
                p, raw_image = filename, raw_images

                if detect is not None and len(detect):
                    # Rescale boxes from img_size to im0 size
                    detect[:, :4] = scale_coords(image.shape[2:], detect[:, :4], raw_image.shape).round()

                    # Print results
                    for category in detect[:, -1].unique():
                        # detections per class
                        number = (detect[:, -1] == category).sum()
                        if number > 1:
                            msg += f"{number} {names[int(category)]}s, "
                        else:
                            msg += f"{number} {names[int(category)]}, "

                    # Write results
                    # Write results
                    for *xyxy, confidence, classes_id in detect:
                        label = f"{names[int(classes_id)]} {int(confidence * 100)}%"
                        plot_one_box(xyxy=xyxy,
                                     image=raw_image,
                                     color=colors[int(classes_id)],
                                     label=label,
                                     line_thickness=3)

                cv2.imwrite(os.path.join(base_path, "new.png"), raw_image)

        context = {
            "status_code": 20000,
            "message": "OK",
            "filename": filename,
            "msg": msg}
        return render(request, "image.html", context)
