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
# Source: https://github.com/pjreddie/darknet/blob/master/scripts/voc_label.py
import os
import xml.etree.ElementTree

from PIL import Image

classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
           'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
           'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
           'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
           'hair drier', 'toothbrush']


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def voc_to_yolo(dataroot, images_dir, annotations_dir, image_index):
    in_file = open(os.path.join(annotations_dir, image_index + ".xml"))
    out_file = open(os.path.join(dataroot, "labels", image_index + ".txt"), "w")
    tree = xml.etree.ElementTree.parse(in_file)
    root = tree.getroot()

    w = 0
    h = 0
    try:
        size = root.find("size")
        w = int(size.find("width").text)
        h = int(size.find("height").text)
    except ValueError:
        pass
    else:
        path = os.path.join("../data/COCO", images_dir, image_index + ".jpg")
        img = Image.open(path)
        w, h = img.size

    for obj in root.iter("object"):
        difficult = obj.find("difficult").text
        cls = obj.find("name").text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find("bndbox")
        box = (float(xmlbox.find("xmin").text),
               float(xmlbox.find("xmax").text),
               float(xmlbox.find("ymin").text),
               float(xmlbox.find("ymax").text))

        bbox = convert((w, h), box)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bbox]) + "\n")


def main(dataroot, images_dir, annotations_dir):
    try:
        os.makedirs("labels")
    except OSError:
        pass

    list_file = open(f"{images_dir[:-4]}.txt", "w")
    for image_index in open(
            os.path.join(dataroot, "ImageSets", "Main", images_dir[:-4] + ".txt")).read().strip().split():
        list_file.write(f"data/{dataroot}/{images_dir.split('/')[:-1]}/{image_index}.jpg\n")
        voc_to_yolo(dataroot, images_dir, annotations_dir, image_index)
    list_file.close()


if __name__ == '__main__':
    main("../data/coco2017", "val2017", "../data/coco2017/Annotations/")
