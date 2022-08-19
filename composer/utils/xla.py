# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""
Helper methods for the tpu-xla backend. 
"""


from composer.trainer.trainer import _is_tpu_installed
import torch

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp

    _is_tpu_installed = True
except ImportError:
    _is_tpu_installed = False


def get_local_rank() -> int:
    return xm.get_local_ordinal()

def get_world_size() -> int:
    return xm.xrt_world_size()

def get_global_rank() -> int:
    return xm.get_ordinal()

def get_local_world_size() -> int:
    return xm.xrt_world_size() # check

def all_reduce(
    tensor: torch.Tensor,
    reduce_operation: str = 'SUM',
) -> None:
    reduce_operation = 'sum'
    xm.all_reduce(reduce_operation, [tensor])
    return tensor


