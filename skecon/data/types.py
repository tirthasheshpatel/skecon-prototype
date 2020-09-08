import numpy as np
import torch

from skecon.utils import _preprocess_data

__all__ = ["TimeSeries", "CrossSection", "Pool", "Panel"]


class DataTypeMixin:
    ...

class TimeSeries(DataTypeMixin):
    ...

class CrossSection(DataTypeMixin):
    ...

class Pool(DataTypeMixin):
    ...

class Panel(DataTypeMixin):
    ...
