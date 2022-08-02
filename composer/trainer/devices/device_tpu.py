# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""The TPU device used for training."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator, TypeVar, Union

import torch

from composer.core.precision import Precision
from composer.trainer.devices.device import Device, T_nnModule

logger = logging.getLogger(__name__)

__all__ = ["DeviceTPU"]

T_nnModule = TypeVar("T_nnModule", bound=torch.nn.Module)

class DeviceTPU(Device):
    """An extension of :class:`~composer.trainer.devices.device.Device` for TPUs
    
    When running on TPUVMs, you need to `export PJRT_DEVICE=TPU`.
    More details.
    """
    dist_backend = "xla-tpu"
    def __init__(self):
        try:
            import torch_xla.core.xla_model as xm
        except ImportError as e:
            raise MissingConditionalImportError(extra_deps_group='tpu', conda_package='torch_xla[tpuvm]') from e
        
        self._device = xm.xla_device()

    def module_to_device(self, module: T_nnModule) -> T_nnModule:
        return module.to(self._device)

    def tensor_to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self._device)

    def state_dict(self) -> StateDict:
        return {}

    def load_state_dict(self, state: StateDict) -> None:
        if len(state) != 0:
            raise ValueError("TPU device has no state.")
