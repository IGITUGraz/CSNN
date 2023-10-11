import torch
from torch import nn
from function import get_spike_fn, get_reset_fn
from typing import NamedTuple, Optional

def autapse_hook(gradient):
    return gradient.clone().fill_diagonal_(0.0)

class BaseSpikingCell(nn.Module):
    
    def __init__(self, threshold: float = 1.0, dt: float = 1.0,
                 spike_grad: str = "triangle", spike_grad_magnitude: float = 0.3,
                 reset_type: str = "subtraction", **kwargs) -> None:
        self.spike_fn = get_spike_fn(spike_grad)
        self.reset_fn = get_reset_fn(reset_type)
        self.thr = threshold
        self.spike_grad_magnitude = spike_grad_magnitude
        self.dt = dt
        super().__init__(**kwargs)
    
    def init_states(self, batch_size: int, device: Optional[torch.device] = None) -> NamedTuple:
        """Initialized states for the cell

        Args:
            batch_size (int): the batch size
            device (torch.device): memory device location

        Raises:
            NotImplementedError: This function 
            should be implemented in concrete sub-class

        Returns:
            NamedTuple: state
        """
        raise NotImplementedError("This function should not be call from BaseCell instance")

class SensoryContextConcat(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    def forward(self, inputs):
        x, c = inputs
        return torch.concat((x, c), -1)
    
class IgnoreContext(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    def forward(self, inputs):
        x,c = inputs
        return x
class CustomSeq(nn.Sequential):
    """
    We sub-class nn.Sequential to provide 
    a way to retrieve recorded states for each layers.
    """
    def get_recorded_states(self):
        recorded_states = {}
        keys = self._modules.keys()
        for k in keys:
            if hasattr(self._modules[k], "get_recorded_states"):
                v = self._modules[k].get_recorded_states()
            else:
                v = []
            recorded_states[k] = v
        return recorded_states
    
    def get_spike_sums(self):
        spikes_sum = []
        keys = self._modules.keys()
        for k in keys:
            if hasattr(self._modules[k], "spike_sum"):
                v = self._modules[k].spike_sum
                spikes_sum.append(v)
        return spikes_sum