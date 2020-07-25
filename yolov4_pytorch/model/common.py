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
from copy import deepcopy

import torch

ONNX_EXPORT = False


def model_info(model, verbose=False):
    # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        print('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

    try:  # FLOPS
        from thop import profile
        flops = profile(deepcopy(model), inputs=(torch.zeros(1, 3, 64, 64),), verbose=False)[0] / 1E9 * 2
        fs = ', %.1f GFLOPS' % (flops * 100)  # 640x640 FLOPS
    except:
        fs = ''

    print('Model Summary: %g layers, %g parameters, %g gradients%s' % (len(list(model.parameters())), n_p, n_g, fs))


def strip_optimizer(f='weights/best.pt'):  # from utils.utils import *; strip_optimizer()
    # Strip optimizer from *.pt files for lighter files (reduced by 1/2 size)
    x = torch.load(f, map_location=torch.device('cpu'))
    x['optimizer'] = None
    x['model'].half()  # to FP16
    torch.save(x, f)
    print('Optimizer stripped from %s, %.1fMB' % (f, os.path.getsize(f) / 1E6))
