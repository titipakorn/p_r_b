"""
 Copyright (c) 2019 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from model import make_model
import torch
import torch.nn as nn


class VectorCNN:
    """Wrapper class for a network returning a vector"""

    def __init__(self, cfg):
        self.model = make_model(cfg, 751)
        self.model.load_param(cfg.TEST_WEIGHT)
        self.device = "cuda"
        if self.device:
            if torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        self.model.eval()

    def forward(self, img):
        """Performs forward of the underlying network on a given batch"""
        #img = torch.from_numpy(img).float().to(self.device)

        return self.model(img).cpu().numpy()
