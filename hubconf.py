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
"""
    File for accessing YOLOv5 via PyTorch Hub https://pytorch.org/hub/
"""

import torch

from yolov4_pytorch.model import YOLO

dependencies = ['torch', 'yaml']


def create(name, pretrained, channels, classes):
    """Creates a specified YOLOv3/4/5 model

    Arguments:
        name (str): name of model, i.e. 'yolov5s'
        pretrained (bool): load pretrained weights into the model
        channels (int): number of input channels
        classes (int): number of model classes

    Returns:
        pytorch model
    """
    model = YOLO(f'configs/COCO-Detection/{name}.yaml', channels, classes)
    if pretrained:
        state_dict = torch.load(f'{name}.pth')['state_dict']
        state_dict = {k: v for k, v in state_dict.items() if model.state_dict()[k].shape == v.shape}  # filter
        model.load_state_dict(state_dict, strict=False)  # load
        model.float()
    return model


def mobilenentv1(pretrained=False, channels=3, classes=80):
    """mobilenent-v1 model from https://github.com/Lornatang/YOLOv4-PyTorch

    Arguments:
        pretrained (bool): load pretrained weights into the model, default=False
        channels (int): number of input channels, default=3
        classes (int): number of model classes, default=80

    Returns:
        pytorch model
    """
    return create('mobilenentv1', pretrained, channels, classes)


def vgg16(pretrained=False, channels=3, classes=80):
    """vgg16 model from https://github.com/Lornatang/YOLOv4-PyTorch

    Arguments:
        pretrained (bool): load pretrained weights into the model, default=False
        channels (int): number of input channels, default=3
        classes (int): number of model classes, default=80

    Returns:
        pytorch model
    """
    return create('vgg16', pretrained, channels, classes)


def yolov3(pretrained=False, channels=3, classes=80):
    """yolov3 model from https://github.com/Lornatang/YOLOv4-PyTorch

    Arguments:
        pretrained (bool): load pretrained weights into the model, default=False
        channels (int): number of input channels, default=3
        classes (int): number of model classes, default=80

    Returns:
        pytorch model
    """
    return create('yolov3', pretrained, channels, classes)


def yolov3_spp(pretrained=False, channels=3, classes=80):
    """YOLOv3_SPP model from https://github.com/Lornatang/YOLOv4-PyTorch

    Arguments:
        pretrained (bool): load pretrained weights into the model, default=False
        channels (int): number of input channels, default=3
        classes (int): number of model classes, default=80

    Returns:
        pytorch model
    """
    return create('yolov3-spp', pretrained, channels, classes)


def yolov3_tiny(pretrained=False, channels=3, classes=80):
    """YOLOv3_tiny model from https://github.com/Lornatang/YOLOv4-PyTorch

    Arguments:
        pretrained (bool): load pretrained weights into the model, default=False
        channels (int): number of input channels, default=3
        classes (int): number of model classes, default=80

    Returns:
        pytorch model
    """
    return create('yolov3-tiny', pretrained, channels, classes)


def yolov4(pretrained=False, channels=3, classes=80):
    """YOLOv4 model from https://github.com/Lornatang/YOLOv4-PyTorch

    Arguments:
        pretrained (bool): load pretrained weights into the model, default=False
        channels (int): number of input channels, default=3
        classes (int): number of model classes, default=80

    Returns:
        pytorch model
    """
    return create('yolov4', pretrained, channels, classes)


def yolov5_small(pretrained=False, channels=3, classes=80):
    """YOLOv5-small model from https://github.com/ultralytics/yolov5

    Arguments:
        pretrained (bool): load pretrained weights into the model, default=False
        channels (int): number of input channels, default=3
        classes (int): number of model classes, default=80

    Returns:
        pytorch model
    """
    return create('yolov5-small', pretrained, channels, classes)


def yolov5_medium(pretrained=False, channels=3, classes=80):
    """YOLOv5-medium model from https://github.com/ultralytics/yolov5

    Arguments:
        pretrained (bool): load pretrained weights into the model, default=False
        channels (int): number of input channels, default=3
        classes (int): number of model classes, default=80

    Returns:
        pytorch model
    """
    return create('yolov5-medium', pretrained, channels, classes)


def yolov5_large(pretrained=False, channels=3, classes=80):
    """YOLOv5-large model from https://github.com/ultralytics/yolov5

    Arguments:
        pretrained (bool): load pretrained weights into the model, default=False
        channels (int): number of input channels, default=3
        classes (int): number of model classes, default=80

    Returns:
        pytorch model
    """
    return create('yolov5-large', pretrained, channels, classes)


def yolov5_xlarge(pretrained=False, channels=3, classes=80):
    """YOLOv5-xlarge model from https://github.com/ultralytics/yolov5

    Arguments:
        pretrained (bool): load pretrained weights into the model, default=False
        channels (int): number of input channels, default=3
        classes (int): number of model classes, default=80

    Returns:
        pytorch model
    """
    return create('yolov5-xlarge', pretrained, channels, classes)
