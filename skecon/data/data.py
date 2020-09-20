import copy

import numpy as np
import torch

from skecon.utils import astensor
from skecon.data.scales import Ratio, Interval, Ordinal, Nominal

class Data:
    SCALES = {"ratio", "interval", "ordinal", "nominal"}

    def __init__(self, data, col_scales=None, dtype=None, device=None, requires_grad=True):
        self.data = astensor(data, dtype=dtype, device=device, requires_grad=requires_grad)
        self.num_cols = self.data.shape[-1]
        self.col_scales = self._process_parameters(col_scales, data)
