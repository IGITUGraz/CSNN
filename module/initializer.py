import torch
from torch.nn.init import _calculate_fan_in_and_fan_out  # noqa
from torch import Tensor


def center_fluctuation_init_(
    tensor: Tensor,
    tau: Tensor,
    alpha: float = 1.0,
    rate: float = 100.0,
):
    with torch.no_grad():
        #tau is fan_out vector length corresponding to 
        epsi_hat = (1e-3*tau) / 2.0
        fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
        std = torch.sqrt(alpha*(1/(fan_in*rate*epsi_hat)))
        std = std.repeat((fan_in, 1)).T
        new_weight = torch.normal(mean=0.0, std=std, out=tensor)
    return new_weight
