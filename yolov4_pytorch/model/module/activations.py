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
    From some excellent activation functions in recent years.
"""
import torch
import torch.nn.functional as F
import torch.nn as nn


class HardSwish(nn.Module):  # https://arxiv.org/pdf/1905.02244.pdf
    @staticmethod
    def forward(x):
        return x * F.hardtanh(x + 3, 0., 6., True) / 6.


class MemoryEfficientMish(nn.Module):
    @staticmethod
    def forward(x):
        return MishImplementation.apply(x)


class MemoryEfficientSwish(nn.Module):
    @staticmethod
    def forward(x):
        return SwishImplementation.apply(x)


class MishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.mul(torch.tanh(F.softplus(x)))  # x * tanh(ln(1 + exp(x)))

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        sx = torch.sigmoid(x)
        fx = F.softplus(x).tanh()
        return grad_output * (fx + x * sx * (1 - fx * fx))


class Mish(nn.Module):  # https://github.com/digantamisra98/Mish
    @staticmethod
    def forward(x):
        return x * F.softplus(x).tanh()


class Swish(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * torch.sigmoid(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        sx = torch.sigmoid(x)
        return grad_output * (sx * (1 + x * (1 - sx)))
