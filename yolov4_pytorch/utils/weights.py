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

import torch
import torch.nn as nn


class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        y = torch.stack(y).mean(0)  # mean ensemble
        return y, None  # inference, train output


def create_pretrained(source='weights/model_best.pth', target='weights/pretrained.pth'):
    x = torch.load(source, map_location=torch.device('cpu'))
    x['optimizer'] = None
    x['epoch'] = -1
    x['state_dict'].half()  # to FP16
    for p in x['state_dict'].parameters():
        p.requires_grad = True
    torch.save(x, target)
    print(f"{source} saved as pretrained checkpoint {target}, {os.path.getsize(target) / 1E6:.1f}MB")


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True
