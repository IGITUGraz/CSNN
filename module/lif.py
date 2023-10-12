import torch
from typing import NamedTuple, Union, Optional
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from .initializer import center_fluctuation_init_
from .base_cell import BaseSpikingCell, autapse_hook
from .li import LILayer

class LIFState(NamedTuple):
    vs: torch.Tensor
    z: torch.Tensor

class LIFCell(BaseSpikingCell):
    def __init__(self, 
                 in_features: int, out_features: int,
                 tau_range:  Union[float, tuple[float, float]] = 20,
                 input_rate: float = 1,
                 threshold: float = 1, dt: float = 1, spike_grad: str = "triangle", 
                 spike_grad_magnitude: float = 0.3, reset_type: str = "subtraction",
                 use_bias: bool = True,
                 use_recurrent: bool = True,
                 device=None,
                 dtype=None,
                 **kwargs) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(threshold, dt, spike_grad, spike_grad_magnitude, reset_type, **kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.tau_range = tau_range
        self.input_rate = input_rate
        self.use_bias = use_bias
        self.use_recurrent = use_recurrent
        
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if self.use_recurrent:
            self.rec_weight = Parameter(torch.empty((out_features, out_features), **factory_kwargs))
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
            # tau is assumed to be a min-max range 
            # from which we uniformly sample
            torch.nn.init.uniform_(self.tau,
                                   torch.asarray(self.tau_range[0]),
                                   torch.asarray(self.tau_range[1]))
        self.decay = torch.exp(-self.dt / self.tau)
        center_fluctuation_init_(self.weight, self.tau, alpha=1 if not self.use_recurrent else 0.8, rate=self.input_rate)
        
        if self.use_recurrent:
            center_fluctuation_init_(
                self.weight, self.tau, 
                alpha=0.2, 
                rate=self.input_rate)
            
            with torch.no_grad():
                self.rec_weight.fill_diagonal_(0.0)
                self.rec_weight.register_hook(autapse_hook)
                
        if self.use_bias:
            torch.nn.init.zeros_(self.bias)
            
    def init_states(self, batch_size: int, device: Optional[torch.device] = None) -> LIFState:
        size = (batch_size, self.out_features)
        vs = torch.zeros(
            size=size,
            device=device,
            dtype=torch.float,
            requires_grad=True
        )
        z = torch.zeros(
            size=size,
            device=device,
            dtype=torch.float,
            requires_grad=True
        )
        state = LIFState(vs=vs, z=z)
        return state
    
    def forward(self, input_tensor: torch.Tensor, state: LIFState) -> tuple[torch.Tensor, LIFState]:
        i_t = F.linear(input_tensor, self.weight, self.bias)
        if self.use_recurrent:
            i_t += F.linear(state.z, self.rec_weight, None)
        vs_tm1 = self.reset_fn(state.vs, state.z, self.thr)
        vs_t = self.decay * vs_tm1 + (1 - self.decay) * i_t
        z_t = self.spike_fn(vs_t - self.thr, self.spike_grad_magnitude)
        return z_t, LIFState(vs=vs_t, z=z_t)

class LIFLayer(LILayer):
    def __init__(self, 
                 in_features: int, out_features: int,
                 tau_range:  Union[float, tuple[float, float]] = 20,
                 input_rate: float = 1,
                 threshold: float = 1, dt: float = 1, spike_grad: str = "triangle", 
                 spike_grad_magnitude: float = 0.3, reset_type: str = "subtraction",
                 use_bias: bool = True,
                 use_recurrent: bool = True,
                 device=None,
                 dtype=None,
                 **kwargs) -> None:
        super().__init__(in_features, out_features, 
                         tau_range, dt, use_bias, device, dtype,**kwargs)
        self.cell = LIFCell(in_features, out_features, tau_range, input_rate, threshold,
                            dt, spike_grad, spike_grad_magnitude,
                            reset_type, use_bias,
                            use_recurrent, device,
                            dtype, **kwargs)
        self.spike_sum = None
        
    def forward(self, input_tensor: torch.Tensor):
        batch_size = input_tensor.shape[0]
        device = input_tensor.device
        state = self.cell.init_states(batch_size, device)
        if self.record_states:
            with torch.no_grad():
                self.states_list = [state]
        outputs = []
        self.spike_sum = state.z.clone()
        for x in input_tensor.unbind(1):
            out, state = self.cell.forward(x, state)
            outputs.append(out)
            self.spike_sum += state.z
            if self.record_states:
                with torch.no_grad():
                    self.states_list.append(state)
            
        outputs = torch.stack(outputs, dim=1)
        return outputs