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

f = open('../data/coco2017/ImageSets/Main/val.txt', 'w')
for filename in os.listdir("../data/coco2017/val_annotations"):
    print(f"Process: {os.path.join('../data/coco2017/val_annotations', filename)}")
    out_path = "../data/coco2017/val_annotations" + filename.replace('xml', 'jpg')
    out_path = out_path.replace('../data/coco2017/val_annotations', '../data/coco2017/JPEGImages/val2017')
    f.write(out_path + '\n')
f.close()
