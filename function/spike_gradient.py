import torch
from typing import Callable

class SuperSpike(torch.autograd.Function):  # noqa
    r"""SuperSpike surrogate gradient as described in Section 3.3.2 of

    F. Zenke, S. Ganguli, **"SuperSpike: Supervised Learning in Multilayer Spiking Neural Networks"**,
    Neural Computation 30, 1514–1541 (2018),
    `doi:10.1162/neco_a_01086 <https://www.mitpressjournals.org/doi/full/10.1162/neco_a_01086>`_
    """

    @staticmethod
    @torch.jit.ignore
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return torch.heaviside(x, torch.as_tensor(0.0).type(x.dtype))

    @staticmethod
    @torch.jit.ignore
    def backward(ctx, grad_output):
        (inp,) = ctx.saved_tensors
        alpha = ctx.alpha
        grad_input = grad_output.clone()
        grad = grad_input / (alpha * torch.abs(inp) + 1.0).pow(
            2
        )  # section 3.3.2 (beta -> alpha)
        return grad, None

class Triangle(torch.autograd.Function):   # noqa

    @staticmethod
    @torch.jit.ignore
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return torch.heaviside(x, torch.as_tensor(0.0).type(x.dtype))

    @staticmethod
    @torch.jit.ignore
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        grad_input = grad_output.clone()
        grad = grad_input * alpha * torch.relu(1 - x.abs())
        return grad, None
    
class Gaussian(torch.autograd.Function):   # noqa

    @staticmethod
    @torch.jit.ignore
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.alpha = torch.as_tensor(alpha).type(x.dtype)
        return torch.heaviside(x, torch.as_tensor(0.0).type(x.dtype))

    @staticmethod
    @torch.jit.ignore
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        grad_input = grad_output.clone()
        grad = grad_input * torch.rsqrt(
            alpha*torch.as_tensor(2.0).type(alpha.dtype)*torch.pi)*torch.exp(-x**2/alpha)
        return grad, None


spike_fn_map = {
    "triangle": Triangle.apply,
    "super": SuperSpike.apply,
    "gaussian": Gaussian.apply
}

def get_spike_fn(method: str) -> Callable[[torch.Tensor, float], torch.Tensor]:
    """
    Exception catching function for clearer error message.
    """
    try:
        return spike_fn_map[method]
    except KeyError:
        print("Attempted to retrieve spike gradient function {method}, but no such "
              + "function exist. We currently support [triangle, super, gaussian].")
