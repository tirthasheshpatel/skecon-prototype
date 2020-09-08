import numpy as np
import torch

from skecon.utils import _preprocess_data

__all__ = ["Ratio", "Interval", "Ordinal", "Nominal", "DataScaleMixin"]

class DataScaleMixin:
    def __init__(self, data, dtype=None, device=None, requires_grad=True):
        self._data = _preprocess_data(data, dtype=dtype,
                                      device=device, requires_grad=requires_grad)
        self._dtype = self._data.dtype
        self._device = self._data.device
        self._requires_grad = requires_grad
        self._processed_data = None

    @property
    def data(self):
        return self._data

    @property
    def processed_data(self):
        return self._processed_data

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._device

    @property
    def requires_grad(self):
        return self._requires_grad

    @data.setter
    def data(self, data):
        if self._processed_data is not None:
            self._processed_data = None
        self._data = _preprocess_data(data, dtype=self.dtype, device=self.device,
                                      requires_grad=self.requires_grad)

    @dtype.setter
    def dtype(self, dtype):
        self._dtype = dtype
        if self._processed_data is not None:
            self._processed_data = torch.as_tensor(self._processed_data, dtype=dtype)
        self._data = torch.as_tensor(self._data, dtype=dtype)

    @device.setter
    def device(self, device):
        self._device = device
        if self._processed_data is not None:
            self._processed_data.to(device)
        self._data.to(device)

    @requires_grad.setter
    def requires_grad(self, requires_grad):
        self._requires_grad = requires_grad
        if requires_grad == False:
            if self._processed_data is not None:
                self._processed_data.detach()
            self._data.detach()
        else:
            if self._processed_data is not None:
                # We need to recompute the 'processed_data' tensor
                # for the transform operations to be in the graph
                self._processed_data = None
            self._data.requires_grad = True

    def _transform_fn(self, *args, **kwargs):
        pass

    def _reverse_transform_fn(self, *args, **kwargs):
        pass

    def transform(self, *args, **kwargs):
        return self._transform_fn(*args, **kwargs)

    def reverse_transform(self, *args, **kwargs):
        return self._reverse_transform_fn(*args, **kwargs)

class Ratio(DataScaleMixin):
    ...

class Interval(DataScaleMixin):
    ...

class Ordinal(DataScaleMixin):
    ...

class Nominal(DataScaleMixin):
    ...
