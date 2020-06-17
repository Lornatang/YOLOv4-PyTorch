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
import shutil
import xml.etree.ElementTree
from pathlib import Path

import cv2
from PIL import Image
from pycocotools.coco import COCO
""
raw_data_dir = "../data/COCO"
data_sets = ["train", "val"]

classes_names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
                 "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                 "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
                 "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
                 "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
                 "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
                 "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
                 "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
                 "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

xml_head = """\
<annotation>
    <folder>VOC</folder>
    <filename>%s</filename>
    <source>
        <database>My Database</database>
        <annotation>COCO</annotation>
        <image>flickr</image>
        <flickrid>NULL</flickrid>
    </source>
    <owner>
        <flickrid>NULL</flickrid>
        <name>company</name>
    </owner>
    <size>
        <width>%d</width>
        <height>%d</height>
        <depth>%d</depth>
    </size>
    <segmented>0</segmented>
"""
xml_object = """\
    <object>
        <name>%s</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>%d</xmin>
            <ymin>%d</ymin>
            <xmax>%d</xmax>
            <ymax>%d</ymax>
        </bndbox>
    </object>
"""
xml_tail = """\
</annotation>
"""


def create_folder(root):
    try:
        os.makedirs(root)
    except OSError:
        print(f"Folder: `{root}` is exist! Do create again!")
        pass


def create_objects(dataset, image_ids=None, category_names=None, category_ids=None):
    annotations_index = dataset.getAnnIds(imgIds=image_ids["id"], catIds=category_ids, iscrowd=None)
    object_bbox = []
    for annotation in dataset.loadAnns(annotations_index):
        category_id = category_names[annotation["category_id"]]
        if category_id in category_names:
            if "bbox" in annotation:
                bbox = annotation["bbox"]
                xmin = int(bbox[0])
                ymin = int(bbox[1])
                xmax = int(bbox[2] + bbox[0])
                ymax = int(bbox[3] + bbox[1])
                object_bbox.append([category_id, xmin, ymin, xmax, ymax])

    return object_bbox


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


def voc_to_yolo(voc_dataroot, voc_images_dir, voc_annotations_dir, voc_image_index):
    # print(f"Process: {os.path.join(annotations_dir, image_index + ".xml")}")
    in_file = open(os.path.join(voc_annotations_dir, voc_image_index + ".xml"))
    out_file = open(os.path.join(voc_dataroot, "labels", voc_image_index + ".txt"), "w")
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
        img = Image.open(os.path.join(raw_data_dir, voc_images_dir, voc_image_index + ".jpg"))
        w, h = img.size

    for obj in root.iter("object"):
        difficult = obj.find("difficult").text
        cls = obj.find("name").text
        if cls not in classes_names or int(difficult) == 1:
            continue
        cls_id = classes_names.index(cls)
        xml_box = obj.find("bndbox")
        box = (float(xml_box.find("xmin").text),
               float(xml_box.find("xmax").text),
               float(xml_box.find("ymin").text),
               float(xml_box.find("ymax").text))

        bbox = convert((w, h), box)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bbox]) + "\n")

    in_file.close()
    out_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Quickly convert COCO format dataset to YOLO format.\n"
                                     "Example: \n\tpython coco2yolo.py --dataroot ../data/coco2017 ")
    parser.add_argument("--dataroot", type=str, default="../data/coco2017",
                        help="Generated dataset address (default: `../data/coco2017`).")
    args = parser.parse_args()

    dataroot = args.dataroot

    for image_dir in data_sets:
        images_dir = image_dir + dataroot[-4:]

        # ------------------------------ COCO convert to Pascal VOC ------------------------------ #
        print(f"=>>>>>> Start COCO {images_dir} dataset format convert to Pascal-VOC dataset format.")
        create_folder(dataroot)
        create_folder(os.path.join(args.dataroot, "JPEGImages"))  # create images dir
        create_folder(os.path.join(dataroot, "Annotations"))

        annotations_file = os.path.join(raw_data_dir, "annotations", f"instances_{images_dir}.json")
        assert Path(annotations_file).exists(), "Annotation file does not exist!"
        coco = COCO(annotations_file)

        # id to name
        names = dict()
        for classes_index in coco.dataset["categories"]:
            names[classes_index["id"]] = classes_index["name"]

        classes_ids = coco.getCatIds(catNms=classes_names)
        for classes in classes_names:
            classes_id = coco.getCatIds(catNms=[classes])
            images_index = coco.getImgIds(catIds=classes_id)
            for image_index in images_index:
                image = coco.loadImgs(image_index)[0]
                filename = image["file_name"]
                objects = create_objects(coco, image, names, classes_ids)
                # get annotation file path
                annotation_path = os.path.join(dataroot, "Annotations", filename[:-3] + "xml")

                raw_image_path = os.path.join(raw_data_dir, images_dir, filename)

                raw_image = cv2.imread(raw_image_path)
                shutil.copy(raw_image_path, os.path.join(dataroot, f"JPEGImages"))

                # parse XML file
                head = xml_head % (filename, raw_image.shape[1], raw_image.shape[0], raw_image.shape[2])
                tail = xml_tail

                # Write the information extracted from json to xml
                annotation_file = open(annotation_path, "w")
                annotation_file.write(head)
                for object_name in objects:
                    annotation_file.write(xml_object % (object_name[0],
                                                        object_name[1],
                                                        object_name[2],
                                                        object_name[3],
                                                        object_name[4]))
                annotation_file.write(tail)
                annotation_file.close()
                # print(f"Process: `{annotation_path}` done!")

        print("=>>>>>> Start get image file name to txt.")
        txt_file = os.path.join(dataroot, "ImageSets", "Main", images_dir[:-4] + ".txt")
        create_folder(os.path.join(dataroot, "ImageSets", "Main"))

        # Remove the file suffix and save
        f = open(txt_file, "w")
        for filename in os.listdir(os.path.join(dataroot, "Annotations")):
            f.write(filename[:-4] + "\n")
            # print(f"Process: `{os.path.join(annotations_dir, filename)}` done!")
        f.close()

        print(f"##### COCO {images_dir} dataset format convert to Pascal-VOC dataset format end! #####\n")
        # ------------------------------ COCO convert to Pascal VOC ------------------------------ #

        # ------------------------------ Generate txt tags corresponding to xml ------------------------------ #
        print("=>>>>>> Start generate txt tags corresponding to xml.")
        create_folder(os.path.join(args.dataroot, "images"))  # create yolo images dir
        create_folder(os.path.join(args.dataroot, "labels"))  # create yolo labels dir

        list_file = open(os.path.join(dataroot, images_dir[:-4] + ".txt"), "w")
        images = open(os.path.join(dataroot, "ImageSets", "Main", images_dir[:-4] + ".txt"))
        for image_index in images.read().strip().split():
            list_file.write(f"data/{dataroot.split('/')[-1]}/{image_index}.jpg\n")
            voc_to_yolo(dataroot, images_dir, os.path.join(dataroot, "Annotations"), image_index)
        list_file.close()
        images.close()
        print("##### Generate txt tags corresponding to xml end! #####\n")
        # ------------------------------ Generate txt tags corresponding to xml ------------------------------ #

        # ------------------------------ Generate txt tags corresponding to xml ------------------------------ #
        print(f"=>>>>>> Clear {images_dir} temporary files.")
        for filename in os.listdir(os.path.join(dataroot, f"JPEGImages")):
            shutil.copy(filename, os.path.join(dataroot, "images"))
        shutil.rmtree(os.path.join(dataroot, "JPEGImages"))
        shutil.rmtree(os.path.join(dataroot, "Annotations"))
        shutil.rmtree(os.path.join(dataroot, "ImageSets"))
        print(f"##### Clear {images_dir} temporary files done! #####")
        # ------------------------------ Generate txt tags corresponding to xml ------------------------------ #
