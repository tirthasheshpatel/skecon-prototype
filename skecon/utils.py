import copy

import torch

from skecon.config import default_device, default_dtype

@torch.no_grad()
def _preprocess_data(data, dtype=None, device=None, shape=None,
                     allow_sparse=False, make_sparse=False,
                     allow_inf=False, allow_nan=False, copy=False,
                     deepcopy=False, requires_grad=False):
    if device is None: device = default_device
    if dtype  is None: dtype  = default_dtype

    if copy and deepcopy:
        raise ValueError("Both 'copy' and 'deepcopy' can't be "
                         "'True' at the same time!")
    if make_sparse == True and allow_sparse == False:
        raise ValueError("'allow_sparse' must be 'True' to "
                         "make a sparse tensor.")

    data = torch.as_tensor(data, dtype=dtype, device=device)
    if requires_grad == True and data.requires_grad == False:
        data.requires_grad = True
    elif requires_grad == False:
        data = data.detach()

    if copy:
        data = data.copy()
    elif deepcopy:
        data = data.clone()
    data = data.to(device).type(dtype)

    if make_sparse:
        raise NotImplementedError("Sparse Tensors API is not "
                                  "implemented yet!")
    if torch.isnan(data) and allow_nan == False:
        raise ValueError("Found 'NaN' entries in the tensor!")
    if torch.isinf(data) and allow_inf == False:
        raise ValueError("Found 'inf' or 'ninf' in the tensor!")
    if shape is not None:
        shape = torch.Size(shape)
        if data.shape != shape:
            raise ValueError("Expected tensor of shape {} "
                             "but found {}".format(data.shape, shape))

    return data
