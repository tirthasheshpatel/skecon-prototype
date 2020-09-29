import copy

import numpy as np
import torch

import skecon
from skecon.utils import astensor
from skecon.data.scales import Ratio, Interval, Ordinal, Nominal

class Data:
    SCALES = {"Ratio", "Interval", "Ordinal", "Nominal"}

    def __init__(self, data, col_scales=None, dtype=None,
                 device=None, requires_grad=True, transform=True,
                 lazy=True):
        (self._transformed_data, self._data,
         self._col_scales, self._transform) = \
            self._process_parameters(col_scales, data, dtype, device,
                                     requires_grad, transform, lazy)
        self._num_cols = self._data.shape[-1]
        self._col_scales = col_scales
        self._dtype = dtype
        self._device = device
        self._requires_grad = requires_grad
        self._transform = transform
        self._lazy = lazy
        self._fitted = False

    def _fit_scale(self, data, col_scales, transform, dtype, device,
                   requires_grad):
        ncols = data.shape[-1]
        self._data_scales = []
        _transformed_data = None
        if len(transform) == 1:
            if transform[0] == False:
                return data
            for c in ncols:
                scale_class = getattr(skecon.data.scales,
                                      col_scales[c], None)
                if scale_class is None:
                    raise RuntimeError(f"scale '{col_scales[c]}' is 'None'!")
                scale = scale_class(data[..., c], dtype=dtype, device=device,
                                    requires_grad=requires_grad)
                scale.fit()
                transformed_data = scale.transform(data[..., c])
                if _transformed_data is None:
                    _transformed_data = transformed_data
                _transformed_data = torch.cat([_transformed_data,
                                               transformed_data],
                                              dim=-1)
                self._data_scales.append(scale)
            return _transformed_data
        for c in ncols:
            scale_class = getattr(skecon.data.scales,
                                    col_scales[c], None)
            if scale_class is None:
                raise RuntimeError(f"scale '{col_scales[c]}' is 'None'!")
            scale = scale_class(data[..., c], dtype=dtype, device=device,
                                requires_grad=requires_grad)
            scale.fit()
            if transform:
                transformed_data = scale.transform(data[..., c])
                if _transformed_data is None:
                    _transformed_data = transformed_data
                _transformed_data = torch.cat([_transformed_data,
                                                transformed_data],
                                                dim=-1)
            else:
                if _transformed_data is None:
                    _transformed_data = data[..., c]
                _transformed_data = torch.cat([_transformed_data,
                                                data[..., c:c+1]],
                                                dim=-1)
            self._data_scales.append(scale)
        return _transformed_data


    def _process_parameters(self, col_scales, data, dtype, device,
                            requires_grad, transform, lazy):
        data = astensor(data, dtype=dtype, device=device,
                        requires_grad=requires_grad)
        ncols = data.shape[-1]
        if data.ndim <= 1:
            raise ValueError("the data must at least be two dimensional")
        col_scales = list(col_scales)
        if len(col_scales) != ncols:
            raise ValueError(f"length of 'col_scales' doesn't match the "
                             f" number of columns in the data. found "
                             f"{len(col_scales)} but expected {ncols}.")
        for col_scale in col_scales:
            if col_scale is not None:
                for col_scale in col_scales:
                    if col_scale is not None and col_scale not in self.SCALES:
                        raise ValueError(f"scale '{col_scale}' not registed! "
                                        f"use Data.SCALES.add('{col_scale}') "
                                        f"to register your custom scale!")
        transform = list(transform)
        if lazy:
            return None, data, col_scales, transform
        else:
            transformed_data = self._fit_scale(data, col_scales, transform,
                                               dtype, device, requires_grad)
            return transformed_data, data, col_scales, transform
        return data, data, col_scales, transform
