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
    Convert PyTorch model to ONNX model structure for model deployment.
"""
import argparse

import onnx
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='../../weights/yolov5s.pth', help='weights path')
    parser.add_argument('--image-size', nargs='+', type=int, default=[640, 640], help='image size')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    opt = parser.parse_args()
    print(opt)

    # Parameters
    f = opt.weights.replace('.pth', '.onnx')  # onnx filename
    image = torch.zeros((opt.batch_size, 3, *opt.img_size))  # image size, (1, 3, 320, 192) iDetection

    # Load PyTorch model
    model = torch.load(opt.weights, map_location=torch.device('cpu'))['model']
    model.eval()
    model.fuse()

    # Export to onnx
    model.model[-1].export = True  # set Detect() layer export=True
    _ = model(image)  # dry run
    # output_names=['classes', 'boxes']
    torch.onnx.export(model, image, f, verbose=False, opset_version=11, input_names=['images'], output_names=['output'])

    # Check onnx model
    model = onnx.load(f)  # load onnx model
    onnx.checker.check_model(model)  # check onnx model
    print(onnx.helper.printable_graph(model.graph))  # print a human readable representation of the graph
    print('Export complete. ONNX model saved to %s\nView with https://github.com/lutzroeder/netron' % f)
