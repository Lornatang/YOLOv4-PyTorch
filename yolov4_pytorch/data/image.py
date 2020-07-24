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
import glob
import math
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from .adjust import exif_size
from .common import create_folder
from .common import random_affine
from .pad_resize import letterbox
from ..utils.coords import xywh2xyxy
from ..utils.coords import xyxy2xywh

help_url = "https://github.com/Lornatang/YOLOv4-PyTorch#train-on-custom-dataset"
image_formats = [".bmp", ".jpg", ".jpeg", ".png", ".tif", ".dng"]
video_formats = ['.mov', '.avi', '.mp4', '.mpg', '.mpeg', '.m4v', '.wmv', '.mkv']


class LoadImages:
    """ Use only in the inference phase
    Load the pictures in the directory and convert them to the corresponding format.
    Args:
        dataroot (str): The source path of the dataset.
        image_size (int): Size of loaded pictures. (default:``416``).
    """

    def __init__(self, dataroot, image_size=640):
        path = str(Path(dataroot))  # os-agnostic
        path = os.path.abspath(path)  # absolute path
        if "*" in path:
            files = sorted(glob.glob(path))  # glob
        elif os.path.isdir(path):
            files = sorted(glob.glob(os.path.join(path, "*.*")))  # dir
        elif os.path.isfile(path):
            files = [path]  # files
        else:
            raise Exception(f"ERROR: {path} does not exist")

        images = [x for x in files if os.path.splitext(x)[-1].lower() in image_formats]
        videos = [x for x in files if os.path.splitext(x)[-1].lower() in video_formats]
        image_num, video_num = len(images), len(videos)

        self.image_size = image_size
        self.files = images + videos
        self.files_num = image_num + video_num
        self.video_flag = [False] * image_num + [True] * video_num
        self.mode = "images"
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.capture = None
        assert self.files_num > 0, f"No images or videos found in {path}. Supported formats are:\nimages: {image_formats}\nvideos: {video_formats}"

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.files_num:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = "video"
            ret_val, raw_image = self.capture.read()
            if not ret_val:
                self.count += 1
                self.capture.release()
                # last video
                if self.count == self.files_num:
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, raw_image = self.capture.read()

            self.frame += 1
            print(f"video {self.count + 1}/{self.files_num}"
                  f"({self.frame}/{self.frames_num}) {path}: ", end="")

        else:
            # Read image
            self.count += 1
            raw_image = cv2.imread(path)  # opencv read image default is BGR
            assert raw_image is not None, "Image Not Found `" + path + "`"
            print(f"image {self.count}/{self.files_num} {path}: ", end="")

        # Padded resize operation
        image = letterbox(raw_image, new_shape=self.image_size)[0]

        # BGR convert to RGB (3 x 416 x 416)
        image = image[:, :, ::-1].transpose(2, 0, 1)
        # Return a contiguous array
        image = np.ascontiguousarray(image)

        return path, image, raw_image, self.capture

    def new_video(self, path):
        self.frame = 0
        self.capture = cv2.VideoCapture(path)
        self.frames_num = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.files_num


class LoadImagesAndLabels(Dataset):
    """ Use in training and testing
    Load pictures and labels from the dataset and convert them to the corresponding format.
    Args:
        path (str): Address of the loaded dataset.
        image_size (int, optional): Size of loaded pictures. (default:``416``).
        batch_size (int, optional): How many samples per batch to load. (default: ``16``).
        augment (bool, optional): Whether image enhancement technology is needed. (default: ``False``).
        hyper_parameters (dict, optional): List of super parameters. (default: ``None``).
        rect (bool, optional): Whether to adjust to matrix training. (default: ``False``).
        image_weights (bool, optional): None. (default:``False``).
        cache_images(bool, optional): # cache labels into memory for faster training.
            (WARNING: large dataset may exceed system RAM).(default:``False``).
        single_cls(bool, optional):  Force dataset into single-class mode. (default:``False``).
        pad(float, optional): Represents how many rows / columns are filled in each dimension. (default:``0.0``)
    """

    def __init__(self, path, image_size=640, batch_size=16, augment=False, hyper_parameters=None, rect=False,
                 image_weights=False, cache_images=False, single_cls=False, stride=32, pad=0.0):
        try:
            path = str(Path(path))  # os-agnostic
            parent = str(Path(path).parent) + os.sep
            if os.path.isfile(path):  # file
                with open(path, 'r') as f:
                    f = f.read().splitlines()
                    f = [x.replace('./', parent) if x.startswith('./') else x for x in f]  # local to global path
            elif os.path.isdir(path):  # folder
                f = glob.iglob(path + os.sep + '*.*')
            else:
                raise Exception(f"{path} does not exist")
            self.image_files = [x.replace('/', os.sep) for x in f if os.path.splitext(x)[-1].lower() in image_formats]
        except IOError:
            raise Exception(f"Error loading data from {path}. See {help_url}")

        image_files_num = len(self.image_files)
        assert image_files_num > 0, f"No images found in {path}. See {help_url}"
        batch_index = np.floor(np.arange(image_files_num) / batch_size).astype(np.int)
        batch_num = batch_index[-1] + 1

        self.image_files_num = image_files_num
        self.batch = batch_index  # batch index of image
        self.image_size = image_size
        self.augment = augment
        self.hyper_parameters = hyper_parameters
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        # load 4 images at a time into a mosaic (only during training)
        self.mosaic = self.augment and not self.rect
        self.mosaic_border = [-image_size // 2, -image_size // 2]
        self.stride = stride

        # Define labels
        self.label_files = [x.replace("images", "labels").replace(os.path.splitext(x)[-1], ".txt")
                            for x in self.image_files]

        # Read image shapes (wh)
        sp = path.replace(".txt", ".shapes")  # shapefile path
        try:
            with open(sp, "r") as f:  # read existing shapefile
                s = [x.split() for x in f.read().splitlines()]
                assert len(s) == self.image_files_num, "Shapefile out of sync"
        except:
            s = [exif_size(Image.open(f)) for f in
                 tqdm(self.image_files, desc="Reading image shapes")]
            np.savetxt(sp, s, fmt="%g")  # overwrites existing (if any)

        self.shapes = np.array(s, dtype=np.float64)

        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            aspect_ratio = s[:, 1] / s[:, 0]  # aspect ratio
            irect = aspect_ratio.argsort()
            self.image_files = [self.image_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.shapes = s[irect]  # wh
            aspect_ratio = aspect_ratio[irect]

            # Set training image shapes
            shapes = [[1, 1]] * batch_num
            for i in range(batch_num):
                aspect_ratio_index = aspect_ratio[batch_index == i]
                mini, maxi = aspect_ratio_index.min(), aspect_ratio_index.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * image_size / stride + pad).astype(np.int) * stride

        # Preload labels (required for weighted CE training)
        self.images = [None] * image_files_num
        self.labels = [np.zeros((0, 5), dtype=np.float32)] * image_files_num
        create_datasubset, extract_bounding_boxes, labels_loaded = False, False, False
        nm, nf, ne, ns, nd = 0, 0, 0, 0, 0  # number missing, found, empty, datasubset, duplicate
        np_labels_path = str(Path(self.label_files[0]).parent) + '.npy'  # saved labels in *.npy file
        if os.path.isfile(np_labels_path):
            s = np_labels_path  # print string
            x = np.load(np_labels_path, allow_pickle=True)
            if len(x) == image_files_num:
                self.labels = x
                labels_loaded = True
        else:
            s = path.replace('images', 'labels')

        process_bar = tqdm(self.label_files)
        for i, image_file in enumerate(process_bar):
            if labels_loaded:
                l = self.labels[i]
                # np.savetxt(file, l, '%g')  # save *.txt from *.npy file
            else:
                try:
                    with open(image_file, 'r') as f:
                        l = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
                except:
                    nm += 1  # print(f'missing labels for image {self.image_files[i]}')  # file missing
                    continue

            if l.shape[0]:
                assert l.shape[1] == 5, f'> 5 label columns: {image_file}'
                assert (l >= 0).all(), f'negative labels: {image_file}'
                assert (l[:, 1:] <= 1).all(), f'non-normalized or out of bounds coordinate labels: {image_file}'
                if np.unique(l, axis=0).shape[0] < l.shape[0]:  # duplicate rows
                    nd += 1  # print('WARNING: duplicate rows in %s' % self.label_files[i])  # duplicate rows
                if single_cls:
                    l[:, 0] = 0  # force dataset into single-class mode
                self.labels[i] = l
                nf += 1  # file found

                # Create subdataset (a smaller dataset)
                if create_datasubset and ns < 1E4:
                    if ns == 0:
                        create_folder(path='./datasubset')
                        os.makedirs('./datasubset/images')
                    exclude_classes = 43
                    if exclude_classes not in l[:, 0]:
                        ns += 1
                        # shutil.copy(src=self.img_files[i], dst='./datasubset/images/')  # copy image
                        with open('./datasubset/images.txt', 'a') as f:
                            f.write(self.image_files[i] + '\n')

                # Extract object detection boxes for a second stage classifier
                if extract_bounding_boxes:
                    p = Path(self.image_files[i])
                    img = cv2.imread(str(p))
                    h, w = img.shape[:2]
                    for j, x in enumerate(l):
                        f = '%s%sclassifier%s%g_%g_%s' % (p.parent.parent, os.sep, os.sep, x[0], j, p.name)
                        if not os.path.exists(Path(f).parent):
                            os.makedirs(Path(f).parent)  # make new output folder

                        b = x[1:] * [w, h, w, h]  # box
                        b[2:] = b[2:].max()  # rectangle to square
                        b[2:] = b[2:] * 1.3 + 30  # pad
                        b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(np.int)

                        b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # clip boxes outside of image
                        b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                        assert cv2.imwrite(f, img[b[1]:b[3], b[0]:b[2]]), 'Failure extracting classifier boxes'
            else:
                ne += 1  # print(f'empty labels for image {self.image_files[i]}')  # file empty

            process_bar.desc = 'Caching labels %s (%g found, %g missing, %g empty, %g duplicate, for %g images)' % (
                s, nf, nm, ne, nd, image_files_num)
        assert nf > 0 or image_files_num == 20288, 'No labels found in %s. See %s' % (
            os.path.dirname(image_file) + os.sep, help_url)
        if not labels_loaded and image_files_num > 1000:
            print('Saving labels to %s for faster future loading' % np_labels_path)
            np.save(np_labels_path, self.labels)  # save for next time

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        if cache_images:  # if training
            gb = 0  # Gigabytes of cached images
            process_bar = tqdm(range(len(self.image_files)), desc='Caching images')
            self.image_hw0, self.image_hw = [None] * image_files_num, [None] * image_files_num
            for i in process_bar:  # max 10k images
                self.images[i], self.image_hw0[i], self.image_hw[i] = load_image(self,
                                                                                 i)  # img, hw_original, hw_resized
                gb += self.images[i].nbytes
                process_bar.desc = f'Caching images ({gb / 1E9:.1f}GB)'

        # Detect corrupted images https://medium.com/joelthchao/programmatically-detect-corrupted-image-8c1b2006c3d3
        detect_corrupted_images = False
        if detect_corrupted_images:
            from skimage import io  # conda install -c conda-forge scikit-image
            for image_file in tqdm(self.image_files, desc='Detecting corrupted images'):
                try:
                    _ = io.imread(image_file)
                except:
                    print('Corrupted image detected: %s' % image_file)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        if self.image_weights:
            index = self.indices[index]

        parameters = self.hyper_parameters
        if self.mosaic:
            # Load mosaic
            images, labels = load_mosaic(self, index)
            shapes = None

        else:
            # Load image
            images, (h0, w0), (h, w) = load_image(self, index)

            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.image_size  # final letterboxed shape
            images, ratio, pad = letterbox(images, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            # Load labels
            labels = []
            x = self.labels[index]
            if x.size > 0:
                # Normalized xywh to pixel xyxy format
                labels = x.copy()
                labels[:, 1] = ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]  # pad width
                labels[:, 2] = ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]  # pad height
                labels[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3] / 2) + pad[0]
                labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]

        if self.augment:
            # Augment imagespace
            if not self.mosaic:
                images, labels = random_affine(images, labels,
                                               degrees=self.hyper_parameters['degrees'],
                                               translate=self.hyper_parameters['translate'],
                                               scale=self.hyper_parameters['scale'],
                                               shear=self.hyper_parameters['shear'])

            # Augment colorspace
            augment_hsv(images, hgain=self.hyper_parameters['hsv_h'],
                        sgain=self.hyper_parameters['hsv_s'],
                        vgain=self.hyper_parameters['hsv_v'])

            # Apply cutouts
            # if random.random() < 0.9:
            #     labels = cutout(img, labels)

        nL = len(labels)  # number of labels
        if nL:
            # convert xyxy to xywh
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

            # Normalize coordinates 0 - 1
            labels[:, [2, 4]] /= images.shape[0]  # height
            labels[:, [1, 3]] /= images.shape[1]  # width

        if self.augment:
            # random left-right flip
            lr_flip = True
            if lr_flip and random.random() < 0.5:
                images = np.fliplr(images)
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]

            # random up-down flip
            ud_flip = False
            if ud_flip and random.random() < 0.5:
                images = np.flipud(images)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]

        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        images = images[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        images = np.ascontiguousarray(images)

        return torch.from_numpy(images), labels_out, self.image_files[index], shapes

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes


def load_image(self, index):
    # loads 1 image from dataset, returns image, original hw, resized hw
    image = self.images[index]
    if image is None:  # not cached
        path = self.image_files[index]
        image = cv2.imread(path)  # BGR
        assert image is not None, "Image Not Found " + path
        h0, w0 = image.shape[:2]  # orig hw
        r = self.image_size / max(h0, w0)  # resize image to image_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
            image = cv2.resize(image, (int(w0 * r), int(h0 * r)), interpolation=interp)
        return image, (h0, w0), image.shape[:2]  # image, hw_original, hw_resized
    else:
        return self.images[index], self.image_hw0[index], self.image_hw[index]  # image, hw_original, hw_resized


def load_mosaic(self, index):
    # loads images in a mosaic

    labels4 = []
    s = self.image_size
    yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, ycenter x, y
    indices = [index] + [random.randint(0, len(self.labels) - 1) for _ in range(3)]  # 3 additional image indices
    for i, index in enumerate(indices):
        # Load image
        image, _, (h, w) = load_image(self, index)

        # place img in img4
        if i == 0:  # top left
            image4 = np.full((s * 2, s * 2, image.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        image4[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        x = self.labels[index]
        labels = x.copy()
        if x.size > 0:  # Normalized xywh to pixel xyxy format
            labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw
            labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh
            labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw
            labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh
        labels4.append(labels)

    # Concat/clip labels
    if len(labels4):
        labels4 = np.concatenate(labels4, 0)
        # np.clip(labels4[:, 1:] - s / 2, 0, s, out=labels4[:, 1:])  # use with center crop
        np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])  # use with random_affine

        # Replicate
        # image4, labels4 = replicate(img4, labels4)

    # Augment
    # img4 = img4[s // 2: int(s * 1.5), s // 2:int(s * 1.5)]  # center crop (WARNING, requires box pruning)
    image4, labels4 = random_affine(image4, labels4,
                                    degrees=self.hyper_parameters['degrees'],
                                    translate=self.hyper_parameters['translate'],
                                    scale=self.hyper_parameters['scale'],
                                    shear=self.hyper_parameters['shear'],
                                    border=self.mosaic_border)  # border to remove

    return image4, labels4


def scale_image(image, ratio=1.0, same_shape=False):  # image(16,3,256,416), r=ratio
    # scales img(bs,3,y,x) by ratio
    height, width = image.shape[2:]
    size = (int(height * ratio), int(width * ratio))  # new size
    image = F.interpolate(image, size=size, mode="bilinear", align_corners=False)  # resize
    if not same_shape:  # pad/crop img
        gs = 32  # (pixels) grid size
        height, width = [math.ceil(x * ratio / gs) * gs for x in (height, width)]
    return F.pad(image, [0, width - size[1], 0, height - size[0]], value=0.447)  # value = imagenet mean


def augment_hsv(image, hgain=0.5, sgain=0.5, vgain=0.5):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    dtype = image.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    image_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR, dst=image)  # no return needed

    # Histogram equalization
    # if random.random() < 0.2:
    #     for i in range(3):
    #         image[:, :, i] = cv2.equalizeHist(image[:, :, i])
