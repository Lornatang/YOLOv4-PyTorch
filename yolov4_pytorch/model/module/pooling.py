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
import torch.nn as nn


class Maxpool(nn.Module):
    # Standard max pool
    def __init__(self, kernel_size, stride=None, padding=0):
        super(Maxpool, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.maxpool(x)
