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
import time
from threading import Thread

import cv2
import numpy as np

from .pad_resize import letterbox


class LoadWebCam:
    """ Use only in the inference phase
        Load the Camera in the local and convert them to the corresponding format.
        Args:
            pipe (int): Device index of camera. (default:``0``).
            image_size (int): Size of loaded pictures. (default:``416``).
        """
    def __init__(self, pipe=0, image_size=640):
        self.image_size = image_size

        if pipe == "0":
            pipe = 0  # local camera

        self.pipe = pipe
        self.cap = cv2.VideoCapture(pipe)  # video capture object
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # set buffer size

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == ord("q"):  # q to quit
            self.cap.release()
            cv2.destroyAllWindows()
            raise StopIteration

        # Read frame
        if self.pipe == 0:  # local camera
            ret_val, raw_image = self.cap.read()
            raw_image = cv2.flip(raw_image, 1)  # flip left-right
        else:  # IP camera
            n = 0
            while True:
                n += 1
                self.cap.grab()
                if n % 30 == 0:  # skip frames
                    ret_val, raw_image = self.cap.retrieve()
                    if ret_val:
                        break

        # Print
        assert ret_val, f"Camera Error {self.pipe}"
        image_path = "webcam.png"
        print(f"webcam {self.count}: ", end="")

        # Padded resize
        image = letterbox(raw_image, new_shape=self.image_size)[0]

        # Convert
        image = image[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        image = np.ascontiguousarray(image)

        return image_path, image, raw_image, None

    def __len__(self):
        return 0


class LoadStreams:
    """ For reading camera or network data
    Load data types from data flow.
    Args:
        sources (str): Data flow file name.
        image_size (int): Image size in default data flow. (default:``416``).
    """

    def __init__(self, sources="streams.txt", image_size=640):
        self.mode = "images"
        self.image_size = image_size

        if os.path.isfile(sources):
            with open(sources, "r") as f:
                sources = [x.strip() for x in f.read().splitlines() if len(x.strip())]
        else:
            sources = [sources]

        n = len(sources)
        self.images = [None] * n
        self.sources = sources
        for i, s in enumerate(sources):
            # Start the thread to read frames from the video stream
            print("%g/%g: %s... " % (i + 1, n, s), end="")
            cap = cv2.VideoCapture(0 if s == "0" else s)
            assert cap.isOpened(), "Failed to open %s" % s
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) % 100
            _, self.images[i] = cap.read()  # guarantee first frame
            thread = Thread(target=self.update, args=([i, cap]), daemon=True)
            print(f" success ({w}x{h} at {fps:.2f} FPS).")
            thread.start()
        print("")  # newline

        # check for common shapes
        s = np.stack([letterbox(x, new_shape=self.image_size)[0].shape for x in self.images], 0)  # inference shapes
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            print("WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.")

    def update(self, index, cap):
        # Read next stream frame in a daemon thread
        n = 0
        while cap.isOpened():
            n += 1
            # _, self.images[index] = cap.read()
            cap.grab()
            if n == 4:  # read every 4th frame
                _, self.images[index] = cap.retrieve()
                n = 0
            time.sleep(0.01)  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        raw_image = self.images.copy()
        if cv2.waitKey(1) == ord("q"):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        image = [letterbox(x, new_shape=self.image_size, auto=self.rect)[0] for x in raw_image]

        # Stack
        image = np.stack(image, 0)

        # Convert
        image = image[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        image = np.ascontiguousarray(image)

        return self.sources, image, raw_image, None

    def __len__(self):
        return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years
