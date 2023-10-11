import torch
from typing import Callable, NamedTuple, Union, Optional
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from .base_cell import BaseSpikingCell, autapse_hook
from .li import LILayer

from module.initializer import center_fluctuation_init_


interact_fn_map = {
    "relu_mult": lambda s, a: s*torch.relu(a),
    "tanh_mult": lambda s, a: s*(1 + torch.tanh(a)),
    "add": lambda s, a: s + a,
    "relu_add": lambda s, a: s + torch.relu(a),
    "tanh_add": lambda s, a: s + torch.tanh(a)
}

def get_interact_fn(method: str) -> Callable[[torch.Tensor, float], torch.Tensor]:
    """
    Exception catching function for clearer error message.
    """
    try:
        return interact_fn_map[method]
    except KeyError:
        print(f"Attempted to retrieve interaction function {method}, but no such "
              + "function exist.")


class CLIFState(NamedTuple):
    vs: torch.Tensor
    va: torch.Tensor
    z: torch.Tensor


class CLIFCell(BaseSpikingCell):
    def __init__(self, 
                 in_features: int, ctx_features:int, out_features: int,
                 tau_soma_range:  Union[float, tuple[float, float]] = 20,
                 tau_apical_range: Union[float, tuple[float, float]] = 20,
                 rate_soma: float = 1,
                 rate_apical: float = 1,
                 threshold: float = 1, dt: float = 1, 
                 spike_grad: str = "triangle", 
                 spike_grad_magnitude: float = 0.3, 
                 reset_type: str = "subtraction",
                 interaction_type: str = "relu_mult",
                 use_bias: bool = True,
                 use_soma_recurrent: bool = True,
                 use_apical_recurrent: bool = True,
                 device=None,
                 dtype=None,
                 **kwargs) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(threshold, dt, spike_grad, spike_grad_magnitude, reset_type, **kwargs)
        self.in_features = in_features
        self.out_features = out_features
        
        self.tau_soma_range = tau_soma_range
        self.tau_apical_range = tau_apical_range
        self.rate_soma = rate_soma
        self.rate_apical = rate_apical
        
        self.use_bias = use_bias
        
        self.use_soma_recurrent = use_soma_recurrent
        self.use_apical_recurrent = use_apical_recurrent
        
        self.interact_fn = get_interact_fn(interaction_type)
        
        self.weight_soma = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.weight_apical = Parameter(torch.empty((out_features, ctx_features), **factory_kwargs))
        if self.use_soma_recurrent:
            self.rec_weight_soma = Parameter(torch.empty((out_features, out_features), **factory_kwargs))
        if self.use_apical_recurrent:
            self.rec_weight_apical = Parameter(torch.empty((out_features, out_features), **factory_kwargs))
        if self.use_bias:
            self.bias_soma = Parameter(torch.empty(out_features, **factory_kwargs))
            self.bias_apical = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_buffer("bias_soma", None)
            self.register_buffer("bias_apical", None)
        self.register_buffer("decay_soma", torch.empty(out_features, **factory_kwargs))
        self.register_buffer("decay_apical", torch.empty(out_features, **factory_kwargs))
        self.register_buffer("tau_soma", torch.empty(out_features, **factory_kwargs))
        self.register_buffer("tau_apical", torch.empty(out_features, **factory_kwargs))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        if not isinstance(self.tau_soma_range, tuple):
            torch.nn.init.constant_(self.tau_soma, torch.asarray(self.tau_soma_range))
        else:
            torch.nn.init.uniform_(self.tau_soma, torch.asarray(self.tau_soma_range[0]), torch.asarray(self.tau_soma_range[1]))
        self.decay_soma = torch.exp(-self.dt/self.tau_soma)
        if not isinstance(self.tau_apical_range, tuple):
            torch.nn.init.constant_(self.tau_apical, torch.asarray(self.tau_apical_range))
        else:
            torch.nn.init.uniform_(self.tau_apical, torch.asarray(self.tau_apical_range[0]), torch.asarray(self.tau_apical_range[1]))
        self.decay_apical = torch.exp(-self.dt/self.tau_apical)
        center_fluctuation_init_(self.weight_soma, 
                                 self.tau_soma, 
                                 alpha=1 if not self.use_soma_recurrent else 0.8, 
                                 rate=self.rate_soma)
        center_fluctuation_init_(self.weight_apical, 
                                 self.tau_apical, 
                                 alpha=1 if not self.use_apical_recurrent else 0.8, 
                                 rate=self.rate_apical)
        if self.use_soma_recurrent:
            center_fluctuation_init_(
                self.rec_weight_soma, 
                self.tau_soma, 
                alpha=0.2, 
                rate=self.rate_soma+self.rate_apical)
            with torch.no_grad():
                self.rec_weight_soma.fill_diagonal_(0.0)
                self.rec_weight_soma.register_hook(autapse_hook)
        if self.use_apical_recurrent:
            center_fluctuation_init_(
                self.rec_weight_apical, 
                self.tau_apical, 
                alpha=0.2, 
                rate=self.rate_soma+self.rate_apical)
            with torch.no_grad():
                self.rec_weight_apical.fill_diagonal_(0.0)
                self.rec_weight_apical.register_hook(autapse_hook)        
        if self.use_bias:
            torch.nn.init.zeros_(self.bias_soma)
            torch.nn.init.zeros_(self.bias_apical)
            
    def init_states(self, batch_size: int, device: Optional[torch.device] = None) -> CLIFState:
        size = (batch_size, self.out_features)
        vs = torch.zeros(
            size=size,
            device=device,
            dtype=torch.float,
            requires_grad=True
        )
        va = torch.zeros(
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
        state = CLIFState(vs=vs, va=va, z=z)
        return state
    
    def forward(self, input_tensor: torch.Tensor, state: CLIFState) -> tuple[torch.Tensor, CLIFState]:
        soma_input, apical_input = input_tensor
        is_t = F.linear(soma_input, self.weight_soma, self.bias_soma)
        ia_t = F.linear(apical_input, self.weight_apical, self.bias_apical)

        if self.use_soma_recurrent:
            is_t += F.linear(state.z, self.rec_weight_soma, None)
        if self.use_apical_recurrent:
            ia_t += F.linear(state.z, self.rec_weight_apical, None)
        
        va_t = self.decay_apical * state.va + (1 - self.decay_apical)  * ia_t
        vs_t = self.decay_soma * state.vs + (1 - self.decay_soma) * self.interact_fn(is_t, va_t)
        
        vs_t = self.reset_fn(vs_t, state.z, self.thr)
        z_t = self.spike_fn((vs_t - self.thr) / self.thr, self.spike_grad_magnitude)
        return z_t, CLIFState(vs=vs_t, va=va_t, z=z_t)



class CLIFLayer(LILayer):
    def __init__(self, 
                 in_features: int, ctx_features: int, out_features: int,
                 tau_soma_range:  Union[float, tuple[float, float]] = 20,
                 tau_apical_range:  Union[float, tuple[float, float]] = 20,
                 rate_soma: float = 1,
                 rate_apical: float = 1,
                 threshold: float = 1, dt: float = 1, spike_grad: str = "triangle", 
                 spike_grad_magnitude: float = 0.3, reset_type: str = "subtraction",
                 interaction_type: str = "relu_mult",
                 use_bias: bool = True,
                 use_soma_recurrent: bool = True,
                 use_apical_recurrent: bool = True,
                 device=None,
                 dtype=None,
                 **kwargs) -> None:
        super().__init__(in_features, out_features, 
                         tau_soma_range, rate_soma, dt, use_bias, device, dtype,**kwargs)
        self.cell = CLIFCell(in_features, ctx_features, out_features, tau_soma_range, tau_apical_range,
                             rate_soma, rate_apical,
                             threshold,
                            dt, spike_grad, spike_grad_magnitude,
                            reset_type, 
                            interaction_type, 
                            use_bias,
                            use_soma_recurrent,
                            use_apical_recurrent,
                            device,
                            dtype, **kwargs)
        self.spike_sum = None
    def forward(self, input_tensor: torch.Tensor):
        sensory, context = input_tensor
        batch_size = sensory.shape[0]
        device = sensory.device
        state = self.cell.init_states(batch_size, device)
        if self.record_states:
            with torch.no_grad():
                self.states_list = [state]
        outputs = []
        # use for spike regularization
        self.spike_sum = state.z.clone()
        for x, c in zip(sensory.unbind(1), context.unbind(1)):
            out, state = self.cell.forward((x, c), state)
            self.spike_sum += state.z
            outputs.append(out)
            if self.record_states:
                with torch.no_grad():
                    self.states_list.append(state)
        outputs = torch.stack(outputs, dim=1)
        return outputs