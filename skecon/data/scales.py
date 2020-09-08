import numpy as np
import torch

from skecon.utils import _preprocess_data

__all__ = ["Ratio", "Interval", "Ordinal", "Nominal", "DataScaleMixin"]

class DataScaleMixin:
    def __init__(self, data, transformer, dtype=None, device=None, requires_grad=True):
        self._data = _preprocess_data(data, dtype=dtype,
                                      device=device, requires_grad=requires_grad)
        if transformer is not None and transformer not in self.TRANSFORMERS:
            raise ValueError(f"Transformer {transformer} for scale {self.__class__.__name__} not found!")
        self._transformer = transformer
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

    @property
    def transformer(self):
        return self._transformer

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

    @transformer.setter
    def transformer(self, transformer):
        if (transformer not in self.TRANSFORMERS):
            raise ValueError(f"Transformer {transformer} for scale {self.__class__.__name__} not found!")
        self._transformer = transformer
        if self._processed_data is not None:
            self._processed_data = None

    def transform(self, *args, **kwargs):
        if self._processed_data is not None:
            return self._processed_data
        transformer_fn = getattr(self, "_" + self.transformer)
        return transformer_fn(*args, **kwargs)

    def reverse_transform(self, *args, **kwargs):
        if self._processed_data is None:
            raise ValueError("Call '.transform' method before reverse transform. "
                             "You should also call the transform method again "
                             "after changing the data contained in the scale object.")
        transformer_fn = getattr(self, "_" + self.transformer + "_reverse")
        return transformer_fn(*args, **kwargs)

class Ratio(DataScaleMixin):
    TRANSFORMERS = {'z_score'}

    def _z_score(self):
        self._data_mean = self.data.mean()
        self._data_std = self.data.std()
        self._processed_data = (self.data - self.data.mean()) / self.data.std()
        return self._processed_data

    def _z_score_reverse(self):
        return self._processed_data * self._data_std + self._data_mean

class Interval(DataScaleMixin):
    ...

class Ordinal(DataScaleMixin):
    ...

class Nominal(DataScaleMixin):
    ...
