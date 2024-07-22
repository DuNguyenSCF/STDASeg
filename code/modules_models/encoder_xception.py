"""Each encoder should have following attributes and methods and be inherited from `_base.EncoderMixin`

Attributes:

    _out_channels (list of int): specify number of channels for each encoder feature tensor
    _depth (int): specify number of stages in decoder (in other words number of downsampling operations)
    _in_channels (int): default number of input channels in first Conv2d layer for encoder (usually 3)

Methods:

    forward(self, x: torch.Tensor)
        produce list of features of different spatial resolutions, each feature is a 4D torch.tensor of
        shape NCHW (features should be sorted in descending order according to spatial resolution, starting
        with resolution same as input `x` tensor).

        Input: `x` with shape (1, 3, 64, 64)
        Output: [f0, f1, f2, f3, f4, f5] - features with corresponding shapes
                [(1, 3, 64, 64), (1, 64, 32, 32), (1, 128, 16, 16), (1, 256, 8, 8),
                (1, 512, 4, 4), (1, 1024, 2, 2)] (C - dim may differ)

        also should support number of features according to specified depth, e.g. if depth = 5,
        number of feature tensors = 6 (one with same resolution as input and 5 downsampled),
        depth = 3 -> number of feature tensors = 4 (one with same resolution as input and 3 downsampled).
"""

import torch.nn as nn

from pretrainedmodels.models.torchvision_models import pretrained_settings
from pretrainedmodels.models.xception import Xception

from _base import EncoderMixin

class XceptionEncoder(Xception, EncoderMixin):
    def __init__(self, out_channels, *args, depth=5, **kwargs):
        super().__init__(*args, **kwargs)

        self._out_channels = out_channels
        self._depth = depth
        self._in_channels = 3

        # modify padding to maintain output shape
        self.conv1.padding = (1, 1)
        self.conv2.padding = (1, 1)

        del self.fc

    def make_dilated(self, *args, **kwargs):
        raise ValueError(
            "Xception encoder does not support dilated mode "
            "due to pooling operation for downsampling!"
        )

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(
                self.conv1, self.bn1, self.relu, self.conv2, self.bn2, self.relu
            ),
            self.block1,
            self.block2,
            nn.Sequential(
                self.block3,
                self.block4,
                self.block5,
                self.block6,
                self.block7,
                self.block8,
                self.block9,
                self.block10,
                self.block11,
            ),
            nn.Sequential(
                self.block12, self.conv3, self.bn3, self.relu, self.conv4, self.bn4
            ),
        ]

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict):
        # remove linear
        state_dict.pop("fc.bias", None)
        state_dict.pop("fc.weight", None)

        super().load_state_dict(state_dict)


xception_encoders = {
    "xception": {
        "encoder": XceptionEncoder,
        "pretrained_settings": pretrained_settings["xception"],
        "params": {"out_channels": (3, 64, 128, 256, 728, 2048)},
    }
}