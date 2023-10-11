import torch

def poisson_encode(rate_values: torch.Tensor, seq_length: int,
                   f_max: float = 100, f_min: float = 0.0, 
                   dt: float = 0.001) -> torch.Tensor:
    """
    Encodes a tensor of input values, which are assumed to be in the
    range [0,1] into a tensor of one dimension higher of binary values,
    which represent input spikes.
    See for example https://www.cns.nyu.edu/~david/handouts/poisson.pdf.
    Parameters:
        input_values (torch.Tensor): Input data tensor with values
        assumed to be in the interval [0,1].
        seq_length (int): Number of time steps in the resulting spike train.
        f_max (float): Maximal frequency (in Hertz) which will be emitted.
        dt (float): Integration time step
        (should coincide with the integration time step used in the model)
    Returns:
        A tensor with an extra dimension of size
        `seq_length` containing spikes (1) or no spikes (0).
    """
    return (# noqa
        torch.rand(seq_length, *rate_values.shape,
                   device=rate_values.device).float()
        < dt * (f_max * rate_values + (1 - rate_values)*f_min)
    ).float()
