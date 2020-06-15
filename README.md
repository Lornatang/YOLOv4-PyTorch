# YOLOv4-PyTorch

<p align="center"><img src="assets/bus.jpg" width="640" alt=""></p>
<p align="center"><img src="assets/giraffe.jpg" width="640" alt=""></p>
<p align="center"><img src="assets/zidane.jpg" width="640" alt=""></p>

### Overview
The inspiration for this project comes from [ultralytics/yolov3](https://github.com/ultralytics/yolov3) && [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet) Thanks.

This project is a [YOLOv4](https://arxiv.org/abs/2004.10934) object detection system. Development framework by [PyTorch](https://pytorch.org/).

The goal of this implementation is to be simple, highly extensible, and easy to integrate into your own projects. This implementation is a work in progress -- new features are currently being implemented.  

### Table of contents
1. [About YOLOv4](#about-yolov4)
2. [Installation](#installation)
    * [Clone and install requirements](#clone-and-install-requirements)
    * [Download pre-trained weights](#download-pre-trained-weights)
    * [Download COCO2014](#download-coco2014)
3. [Usage](#usage)
    * [Train](#train)
    * [Test](#test)
    * [Inference](#inference)
4. [Backbone](#backbone)
5. [Train on Custom Dataset](#train-on-custom-dataset)
6. [Darknet Conversion](#darknet-conversion)
7. [Credit](#credit) 

### About YOLOv4
There are a huge number of features which are said to improve Convolutional Neural Network (CNN) accuracy. Practical testing of combinations of such features on large datasets, and theoretical justification of the result, is required. Some features operate on certain models exclusively and for certain problems exclusively, or only for small-scale datasets; while some features, such as batch-normalization and residual-connections, are applicable to the majority of models, tasks, and datasets. We assume that such universal features include Weighted-Residual-Connections (WRC), Cross-Stage-Partial-connections (CSP), Cross mini-Batch Normalization (CmBN), Self-adversarial-training (SAT) and Mish-activation. We use new features: WRC, CSP, CmBN, SAT, Mish activation, Mosaic data augmentation, CmBN, DropBlock regularization, and CIoU loss, and combine some of them to achieve state-of-the-art results: 43.5% AP (65.7% AP50) for the MS COCO dataset at a realtime speed of ~65 FPS on Tesla V100. Source code is at [this https URL](https://github.com/AlexeyAB/darknet).

### Installation

#### Clone and install requirements
```bash
$ git clone https://github.com/Lornatang/YOLOv4-PyTorch.git
$ cd YOLOv4-PyTorch/
$ pip install -r requirements.txt
```

#### Download pre-trained weights
```bash
$ cd weights/
$ bash download_weights.sh
```

#### Download COCO2014
```bash
$ cd data/
$ bash get_coco_dataset.sh