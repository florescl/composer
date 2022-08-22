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
    # check if in-place
    return tensor

def broadcast(tensor: torch.Tensor, src: int) -> None:
    if src != xm.get_ordinal():
        tensor.fill_(0.0)
    xm.all_reduce("sum", [tensor])
    # check if in-place
    return tensor


def get_sampler(dataset: torch.utils.data.Dataset, *, drop_last: bool, shuffle: bool):
    return torch.utils.data.DistributedSampler[int](
        dataset,
	drop_last=drop_last,
	shuffle=shuffle,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
    )

