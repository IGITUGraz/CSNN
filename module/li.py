import torch
from torch import nn
from typing import NamedTuple, Union, Optional
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from function.utils import merge_states

from module.initializer import center_fluctuation_init_


class LIState(NamedTuple):
    vs: torch.Tensor
  
class LICell(nn.Module):
    
    def __init__(self, in_features: int, out_features: int, 
                 tau_range: Union[float, tuple[float, float]] = 20, 
                 input_rate: float = 1,
                 dt: float = 1.0,
                 use_bias: bool = True,
                 device=None,
                 dtype=None,
                 **kwargs,) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.tau_range = tau_range
        self.input_rate = input_rate
        self.dt = dt
        self.use_bias = use_bias
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if self.use_bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_buffer("bias", None)
        self.register_buffer("decay", torch.empty(out_features, **factory_kwargs))
        self.register_buffer("tau", torch.empty(out_features, **factory_kwargs))
        self.reset_parameters()
    
    def reset_parameters(self):
        if not isinstance(self.tau_range, tuple):
            torch.nn.init.constant_(self.tau, torch.asarray(self.tau_range))
        else:
            # tau is assumed to be a min-max range from which we will uniformly sample
            torch.nn.init.uniform_(self.tau, torch.asarray(self.tau[0]), torch.asarray(self.tau[1]))
        self.decay = torch.exp(-self.dt / self.tau)
            
        if self.use_bias:
            torch.nn.init.zeros_(self.bias)
        center_fluctuation_init_(self.weight, self.tau, 1, self.input_rate)
            
    def init_states(self, batch_size: int, device: Optional[torch.device] = None) -> LIState:
        size = (batch_size, self.out_features)
        vs = torch.zeros(
            size=size,
            device=device,
            dtype=torch.float,
            requires_grad=True
        )
        state = LIState(vs=vs)
        return state
    def forward(self, input_tensor: torch.Tensor,
                state: LIState
                ) -> tuple[torch.Tensor, LIState]:
        
        i_t = F.linear(input_tensor, self.weight, self.bias)
        vs_t = self.decay * state.vs + (1 - self.decay) * i_t
        return vs_t, LIState(vs=vs_t)


class LILayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, 
                 tau_range: Union[float, tuple[float, float]] = 20, 
                 input_rate: float = 1,
                 dt: float = 1.0,
                 use_bias: bool = True,
                 device=None,
                 dtype=None,
                 **kwargs,) -> None:
        super().__init__(**kwargs)
        self.cell = LICell(in_features, out_features, tau_range, input_rate,
                           dt, use_bias, device, dtype, **kwargs)
        self.states_list = []
        self.record_states = False
    
    def forward(self, input_tensor: torch.Tensor):
        batch_size = input_tensor.shape[0]
        device = input_tensor.device
        state = self.cell.init_states(batch_size, device)
        if self.record_states:
            with torch.no_grad():
                self.states_list = [state]
        outputs = []
        for x in input_tensor.unbind(1):
            out, state = self.cell.forward(x, state)
            outputs.append(out)
            if self.record_states:
                with torch.no_grad():
                    self.states_list.append(state)
        outputs = torch.stack(outputs, dim=1)
        return outputs
    
    def get_recorded_states(self) -> NamedTuple:
        if self.states_list == []:
            return []
        else:
            return merge_states(self.states_list)