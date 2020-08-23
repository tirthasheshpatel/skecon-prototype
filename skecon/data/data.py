import copy

import numpy as np
import torch

from skecon.utils import _preprocess_data
from skecon.data.scales import Ratio, Interval, Ordinal, Nominal
from skecon.data.types import TimeSeries, CrossSection, Pool, Panel

class Data:
    SCALES = {"ratio", "interval", "ordinal", "nominal"}
    TYPES  = {"time_series", "cross_section", "pool", "panel"}
    # TODO: Implement imputers

    def __init__(self, data, col_types=None, col_scales=None, dtype=None, device=None):
        self.data = _preprocess_data(data, dtype, device)
        self.num_cols = self.data.shape[-1]
        self.col_types, self.col_scales = self._process_parameters(col_types, col_scales, data)
