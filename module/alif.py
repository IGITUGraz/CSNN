import torch
from typing import Callable, NamedTuple, Union, Optional
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from .base_cell import BaseSpikingCell, autapse_hook
from .li import LILayer

from module.initializer import center_fluctuation_init_


class ALIFState(NamedTuple):
    vs: torch.Tensor
    b: torch.Tensor
    z: torch.Tensor


class ALIFCell(BaseSpikingCell):
    def __init__(self, 
                 in_features: int, out_features: int,
                 tau_soma_range:  Union[float, tuple[float, float]] = 20,
                 tau_beta_range: Union[float, tuple[float, float]] = 20,
                 input_rate: float = 1,
                 threshold: float = 1,
                 thr_incr: float = 1.3,
                 dt: float = 1, 
                 spike_grad: str = "triangle",
                 spike_grad_magnitude: float = 0.3, 
                 reset_type: str = "subtraction",
                 use_bias: bool = True,
                 use_recurrent: bool = True,
                 device=None,
                 dtype=None,
                 **kwargs) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(threshold, dt, spike_grad, spike_grad_magnitude, reset_type, **kwargs)
        self.in_features = in_features
        self.out_features = out_features
        
        self.tau_soma_range = tau_soma_range
        self.tau_beta_range = tau_beta_range
        
        self.thr_incr = thr_incr
        
        self.use_bias = use_bias
        self.input_rate = input_rate
        
        self.use_recurrent = use_recurrent        
        
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if self.use_recurrent:
            self.rec_weight = Parameter(torch.empty((out_features, out_features), **factory_kwargs))
        if self.use_bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_buffer("bias", None)
        self.register_buffer("decay_soma", torch.empty(out_features, **factory_kwargs))
        self.register_buffer("decay_beta", torch.empty(out_features, **factory_kwargs))
        self.register_buffer("tau_soma", torch.empty(out_features, **factory_kwargs))
        self.register_buffer("tau_beta", torch.empty(out_features, **factory_kwargs))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        if not isinstance(self.tau_soma_range, tuple):
            torch.nn.init.constant_(self.tau_soma, torch.asarray(self.tau_soma_range))
        else:
            torch.nn.init.uniform_(self.tau_soma, torch.asarray(self.tau_soma_range[0]), torch.asarray(self.tau_soma_range[1]))
        if not isinstance(self.tau_beta_range, tuple):
            torch.nn.init.constant_(self.tau_beta, torch.asarray(self.tau_beta_range))
        else:
            torch.nn.init.uniform_(self.tau_beta, torch.asarray(self.tau_beta_range[0]), torch.asarray(self.tau_beta_range[1]))
        self.decay_soma = torch.exp(-self.dt/self.tau_soma)
        self.decay_beta = torch.exp(-self.dt/self.tau_beta)
        center_fluctuation_init_(self.weight, 
                                 self.tau_soma, 
                                 alpha=1 if not self.use_recurrent else 0.8, 
                                 rate=self.input_rate)
        if self.use_recurrent:
            center_fluctuation_init_(
                self.rec_weight, 
                self.tau_soma, 
                alpha=0.2, 
                rate=self.input_rate)
            with torch.no_grad():
                self.rec_weight.fill_diagonal_(0.0)
                self.rec_weight.register_hook(autapse_hook)
       
        if self.use_bias:
            torch.nn.init.zeros_(self.bias)
            
    def init_states(self, batch_size: int, device: Optional[torch.device] = None) -> ALIFState:
        size = (batch_size, self.out_features)
        vs = torch.zeros(
            size=size,
            device=device,
            dtype=torch.float,
            requires_grad=True
        )
        b = torch.zeros(
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
        state = ALIFState(vs=vs, b=b, z=z)
        return state
    
    def forward(self, input_tensor: torch.Tensor, state: ALIFState) -> tuple[torch.Tensor, ALIFState]:
        is_t = F.linear(input_tensor, self.weight, self.bias)

        if self.use_recurrent:
            is_t += F.linear(state.z, self.rec_weight, None)
        b_t = self.decay_beta * state.b + (1 - self.decay_beta) * state.z
        adapt_thr = self.thr + self.thr_incr * b_t
        vs_t = self.decay_soma * state.vs + (1 - self.decay_soma) * is_t
        vs_t = self.reset_fn(vs_t, state.z, adapt_thr)
        z_t = self.spike_fn((vs_t - adapt_thr) / adapt_thr, self.spike_grad_magnitude)
        return z_t, ALIFState(vs=vs_t, b=b_t, z=z_t)



class ALIFLayer(LILayer):
    def __init__(self, 
                 in_features: int, out_features: int,
                 tau_soma_range:  Union[float, tuple[float, float]] = 20,
                 tau_beta_range: Union[float, tuple[float, float]] = 20,
                 input_rate: float = 1,
                 threshold: float = 1,
                 thr_incr: float = 1.3,
                 dt: float = 1, 
                 spike_grad: str = "triangle",
                 spike_grad_magnitude: float = 0.3, 
                 reset_type: str = "subtraction",
                 use_bias: bool = True,
                 use_recurrent: bool = True,
                 device=None,
                 dtype=None,
                 **kwargs) -> None:
        super().__init__(in_features, out_features, 
                         tau_soma_range, input_rate, dt, use_bias, device, dtype,**kwargs)
        self.cell = ALIFCell(in_features, out_features, 
                             tau_soma_range, tau_beta_range,
                             input_rate, threshold,
                             thr_incr,
                             dt, spike_grad, spike_grad_magnitude,
                             reset_type,
                             use_bias,
                             use_recurrent,
                             device,
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
        # use for spike regularization
        self.spike_sum = state.z.clone()
        for x in input_tensor.unbind(1):
            out, state = self.cell.forward(x, state)
            self.spike_sum += state.z
            outputs.append(out)
            if self.record_states:
                with torch.no_grad():
                    self.states_list.append(state)
        outputs = torch.stack(outputs, dim=1)
        return outputs