










"""This module contains aliases for structured Tensors, improving readability."""


import torch


from curobo.util.logger import log_warn

T_DOF = torch.Tensor
T_BDOF = torch.Tensor
T_BHDOF_float = torch.Tensor
T_HDOF_float = torch.Tensor

T_BValue_float = torch.Tensor
T_BHValue_float = torch.Tensor
T_BValue_bool = torch.Tensor
T_BValue_int = torch.Tensor

T_BPosition = torch.Tensor
T_BQuaternion = torch.Tensor
T_BRotation = torch.Tensor
