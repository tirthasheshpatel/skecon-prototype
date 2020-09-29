from abc import abstractmethod

import numpy as np
import torch

from skecon.utils import astensor

__all__ = ["Ratio", "Interval", "Ordinal", "Nominal", "DataScaleMixin"]

class DataScaleMixin:
    r"""
    Data Scale Base Class.

    Base class of all the data containers present in `skecon`. This class
    is easily extendible for custom data types and transformers. Please
    see the documentation of the methods to implement custom data types
    and transformers.

    Parameters
    ----------
    data : tensor
        The data set to operate on.

    transformer : str
        The transformer to be used to preprocess the underlying data.
        To get the list of available transformers for any data class,
        use ``cls.TRANSFORMERS``. To implement your own transformer
        with this class as the base class, implement transform function
        with the name `_<transformer_name>`, reverse transform function
        as `_<transformer_name>_reverse` and add the `<transformer_name>`
        to a static class variable called ``TRANSFORMERS``. You will
        also need to create a `_<transformer_name>_fit` method to fit the
        transformer on the given data. Two more optional requirements are
        to add `_<transformer_name>_argcheck` and
        `<transformer_name>_reverse_argcheck` for argument validation. If
        these two methods are not implemented, no argument validation will
        be done and output may contain undesired values.

    dtype : torch.DType
        Data type of the data. The data will be coerced to this data type
        without warnings. If ``None``, default data type is used. To change
        the default data type, see `skecon.config.set_default_dtype`.

    device : torch.Device
        Device of the data. The data will be moved to this device
        without warnings. If ``None``, default device is used. To change
        the default device, see `skecon.config.set_default_device`.

    requires_grad : bool
        If ``True``, creates the operation graph and allows torch to create
        gradient functions for the operations performed on the data.

    Methods
    -------
    fit : Fit and initialize the transformer.
    transform : Transform the data.
    reverse_transform : Reverse transform the data.

    Examples
    --------
    To implement a custom data scale, with a custom transformer, use:

    >>> import torch
    >>> from skecon.data import DataScaleMixin
    >>> class MyDataScale(DataScaleMixin):
    ...     TRANSFORMERS = {"z_score"}
    ...     def _z_score(self, data):
    ...         if (getattr(self, "_data_mean", None) == None and
    ...             getattr(self, "_data_std", None) == None):
    ...             self._data_mean = self.data.mean()
    ...             self._data_std = self.data.std()
    ...         self._transformed_data = ((self.data - self._data_mean) /
    ...                                 self._data_std)
    ...         return self._transformed_data
    ...     def _z_score_reverse(self, data):
    ...         return (self.transformed_data * self._data_std +
    ...                 self._data_mean)
    ...

    Now you can use the `MyDataScale` class to fit and transform data
    and to reverse transform it.

    >>> data = torch.tensor([1., 2., 3.])
    >>> scale = MyDataScale(data, transformer='z_score')
    >>> scale.transform(data)
    tensor([-1.,  0.,  1.], grad_fn=<DivBackward0>)
    >>> scale.reverse_transform(data)
    tensor([3., 4., 5.], grad_fn=<AddBackward0>)

    To avoid creating a graph and calculating the gradients, you can
    either pass ``requires_grad=False`` while creating the object:

    >>> scale = MyDataScale(data, transformer='z_score', requires_grad=False)

    or it can be turned off using:

    >>> scale.requires_grad = False

    The latter will detach the ``data`` and ``transformed_data`` tensors from
    the graph without any warning or error. A similar behaviour holds for
    other properties like ``dtype``, ``device`` and ``data``. See the examples
    below:

    >>> # To change the data type use:
    ... scale.dtype = torch.float64
    >>> # To change the device use:
    ... scale.device = 'cuda'

    The transformer can be changed using:

    >>> scale.transformer = 'one_hot'

    The above example will throw an error as ``one_hot`` transformer
    hasn't been implemented and the ``DataScaleMixin`` class is not
    aware of it until it has been added to the ``TRANSFORMERS`` class
    variable.

    Raises
    ------
    ValueError : If the given transformer hasn't been implemented.
    """
    def __new__(cls, data, transformer=None, dtype=None,
                device=None, requires_grad=True):
        if transformer is None:
            transformer = cls.DEFAULT
        if transformer is not None and transformer not in cls.TRANSFORMERS:
            raise ValueError(f"Transformer '{transformer}' for scale "
                             f"'{cls.__class__.__name__}' not found!")

        cls._data = astensor(data, dtype=dtype, device=device,
                              requires_grad=requires_grad)
        cls._transformer = transformer
        cls._dtype = cls._data.dtype
        cls._device = cls._data.device
        cls._requires_grad = requires_grad

        cls.fitted = False

        return cls

    @property
    def data(self):
        r"""Get the data set."""
        return self._data

    @property
    def dtype(self):
        r"""Get the dtype of the data."""
        return self._dtype

    @property
    def device(self):
        r"""Get the device on which computations are performed."""
        return self._device

    @property
    def requires_grad(self):
        return self._requires_grad

    @property
    def transformer(self):
        r"""Get the transformer."""
        return self._transformer

    @data.setter
    def data(self, data):
        self._data = astensor(data, dtype=self.dtype, device=self.device,
                              requires_grad=self.requires_grad)
        self.fitted = False

    @dtype.setter
    def dtype(self, dtype):
        self._dtype = dtype
        self._data = torch.as_tensor(self._data, dtype=dtype)

    @device.setter
    def device(self, device):
        self._device = device
        self._data.to(device)

    @requires_grad.setter
    def requires_grad(self, requires_grad):
        if requires_grad != self._requires_grad:
            self._requires_grad = requires_grad
            if requires_grad == False:
                self._data.detach()
            else:
                self._data.requires_grad = True

    @transformer.setter
    def transformer(self, transformer):
        if transformer != self._transformer:
            if (transformer is not None and
                transformer not in self.TRANSFORMERS):
                raise ValueError(f"Transformer {transformer} for scale "
                                f"{self.__class__.__name__} not found!")
            self._transformer = transformer
            self.fitted = False

    def fit(self, *args, **kwargs):
        if self.fitted == False:
            if self.transformer is not None:
                fit_fn = getattr(self, "_" + self.transformer + "_fit", None)
                if fit_fn is not None:
                    fit_fn(*args, **kwargs)
        self.fitted = True

    def transform(self, data, *args, **kwargs):
        r"""
        Transform the data using the `transformer`.

        Parameters
        ----------
        data : tensor
            The data to be transformed.

        *args, **kwargs : Any
            Other arguments and keyword arguments that the
            transformer function accepts.

        Returns
        -------
        transformed_data : tensor
            The transformed data.

        Raises
        ------
        ValueError : If the transformer is not initialized.

        Notes
        -----
        Note that if the ``transformer`` is ``None``, the data will not
        be transformed and returned as is. Invalid instances will be
        masked with ``nan``.
        """
        data = astensor(data, dtype=self.dtype, device=self.device,
                        requires_grad=self.requires_grad)
        if self.transformer is not None:
            transformer_fn = getattr(self, "_" + self.transformer, None)
        else:
            return data
        if self.fitted == False:
            raise ValueError("The transformer is not initialized. "
                             "Call the '.fit' method to initialize "
                             "the transformer.")
        mask_fn = getattr(self, "_" + self.transformer + "_argcheck", None)
        if mask_fn is None:
            return transformer_fn(data, *args, **kwargs)
        with torch.no_grad():
            mask_cond = mask_fn(data, *args, **kwargs)
        result = transformer_fn(data, *args, **kwargs)
        with torch.no_grad():
            mask_cond = ((mask_cond) |
                        (torch.zeros_like(result, dtype=torch.bool)))
            result[mask_cond] = np.nan
        return result

    def reverse_transform(self, data, *args, **kwargs):
        r"""
        Reverse transform the data using the `transformer`.

        Parameters
        ----------
        data : tensor
            The data to be reverse transformed.

        *args, **kwargs : Any
            Other arguments and keyword arguments that the
            transformer function accepts.

        Returns
        -------
        rtransformed_data : tensor
            The reverse transformed data.

        Raises
        ------
        ValueError : If the transformer is not initialized.

        Notes
        -----
        Note that if the ``transformer`` is ``None``, the data will not
        be reverse transformed and returned as is. Invalid instances
        will be masked with ``nan``.
        """
        data = astensor(data, dtype=self.dtype, device=self.device,
                        requires_grad=self.requires_grad)
        if self.transformer is not None:
            rtransformer_fn = getattr(self,
                                      "_" + self.transformer + "_reverse",
                                      None)
        else:
            return data
        if self.fitted == False:
            raise ValueError("The transformer is not initialized. "
                             "Call the '.fit' method to initialize "
                             "the transformer.")
        mask_fn = getattr(self, ("_" + self.transformer +
                                 "_reverse_argcheck"), None)
        if mask_fn is None:
            return rtransformer_fn(data, *args, **kwargs)
        with torch.no_grad():
            mask_cond = mask_fn(data, *args, **kwargs)
        result = rtransformer_fn(data, *args, **kwargs)
        with torch.no_grad():
            mask_cond = ((mask_cond) |
                        (torch.zeros_like(result, dtype=torch.bool)))
            result[mask_cond] = np.nan
        return result

class Ratio(DataScaleMixin):
    # Registered transformers
    TRANSFORMERS = {'z_score'}
    DEFAULT = 'z_score'

    def _z_score_fit(self):
        if self.data.ndim == 0:
            self._data_mean = torch.tensor([],
                                           dtype=self.dtype,
                                           device=self.device,
                                           requires_grad=self.requires_grad)
            self._data_std = self._data_mean
        elif self.data.ndim == 1:
            self._data_mean = self.data.mean(axis=0, keepdims=True)
            self._data_std = self.data.std(axis=0, keepdims=True)
        else:
            self._data_mean = self.data.mean(axis=self.data.ndim-2,
                                            keepdims=True)
            self._data_std = self.data.std(axis=self.data.ndim-2,
                                        keepdims=True)

    def _z_score_argcheck(self, data):
        return ((torch.isnan(self._data_mean)) |
                (torch.isnan(self._data_std)) |
                (self._data_std == 0))

    def _z_score_reverse_argcheck(self, data):
        return self._z_score_argcheck(data)

    def _z_score(self, data):
        self._transformed_data = ((data - self._data_mean) /
                                  self._data_std)
        return self._transformed_data

    def _z_score_reverse(self, data):
        return data * self._data_std + self._data_mean

class Interval(DataScaleMixin):
    ...

class Ordinal(DataScaleMixin):
    ...

class Nominal(DataScaleMixin):
    ...
