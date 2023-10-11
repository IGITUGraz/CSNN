from torch.utils.data import Dataset
import math
import numpy as np
from typing import Optional
import torch
def triangle_wave(nb_sample, p, shift, a):
    i = math.floor(math.log2(p*nb_sample/4))
    dt = p/4 * 1/2**i
    #nb_sample = int(math.floor((t_max - t_min) / dt))
    t_max = nb_sample*dt
    t = np.linspace(dt*shift, t_max + dt*shift, nb_sample, endpoint=False)
    y = (4.0*a)/p * np.abs(((t - p/4) % p) - p/2) - a
    return y
    
def sawthooth_wave(nb_sample, p, shift, a):
    i = math.floor(math.log2(p*nb_sample/3))
    dt = p/3* 1/2**i
    t_max = nb_sample*dt
    t = np.linspace(dt*shift, t_max + dt*shift, nb_sample, endpoint=False)
    y = a*2*(t/p - np.floor(0.5 + t/p))
    return y

def square_wave(nb_sample, p, shift, a):
    i = math.floor(math.log2(p*nb_sample/6))
    dt = p/6 * 1/2**i
    t_max = nb_sample*dt
    t = np.linspace(dt*shift, t_max + dt*shift, nb_sample, endpoint=False)
    y = a*np.sign(np.sin((2*np.pi*t)/p))
    return y

def sin_wave(nb_sample, p, shift, a):
    i = math.floor(math.log2(p*nb_sample/4))
    dt = p/4 * 1/2**i
    t_max = nb_sample*dt
    t = np.linspace(dt*shift, t_max + dt*shift, nb_sample, endpoint=False)
    y = a*np.sin((2*np.pi*t)/p)
    return y


    
class WaveRegression(Dataset):
    def __init__(self, per_class_example: int, 
                 time_steps: int,
                 context_freq: float, dt: float,
                 context_time_steps: Optional[int] = None) -> None:
        self.per_class_example = per_class_example
        self.time_steps = time_steps
        self.context_freq = context_freq
        self.dt = dt
        if context_time_steps is not None:
            assert context_time_steps <= time_steps, "Context time steps cannot be longer"\
                "than the total number of time steps in the sequence."
            self.context_time_steps = context_time_steps
        else:
            self.context_time_steps = time_steps

        triangle_phases = np.random.randint(-4, 4, size=(per_class_example,))
        sawthooth_phases = np.random.randint(-3, 3, size=(per_class_example))
        square_phases = np.random.randint(-6, 6, size=(per_class_example,))
        sin_phases = np.random.randint(-4, 4, size=(per_class_example,))
        amplitude = np.random.uniform(0.1, 3, size=(4, per_class_example,))
        # fix the frequency for now, different wave requiered different non-uniform sampling interval at different freq...
        signal_freq = 5
        triangle_wave_data = np.array(
            list(
                map(lambda x: triangle_wave(self.time_steps+1, 1/signal_freq, x[0], x[1]),
                    list(zip(triangle_phases, amplitude[0])))), dtype=np.float_)
        sawthooth_wave_data = np.array(
            list(
                map(lambda x: sawthooth_wave(self.time_steps+1, 1/signal_freq, x[0], x[1]),
                    list(zip(sawthooth_phases, amplitude[1])))), dtype=np.float_)
        square_wave_data = np.array(
            list(
                map(lambda x: square_wave(self.time_steps+1, 1/signal_freq, x[0], x[1]),
                    list(zip(square_phases, amplitude[2])))), dtype=np.float_)
        sin_wave_data = np.array(
            list(
                map(lambda x: sin_wave(self.time_steps+1, 1/signal_freq, x[0], x[1]),
                    list(zip(sin_phases, amplitude[3])))), dtype=np.float_)
        
        self.data = torch.asarray(
            np.concatenate([
                triangle_wave_data[:, :-1],
                sawthooth_wave_data[:, :-1],
                square_wave_data[:, :-1],
                sin_wave_data[:, :-1]], axis=0), dtype=torch.float).unsqueeze(-1)
        self.target = torch.asarray(
            np.concatenate([
                triangle_wave_data[:, 1:],
                sawthooth_wave_data[:, 1:],
                square_wave_data[:, 1:],
                sin_wave_data[:, 1:]], axis=0), dtype=torch.float).unsqueeze(-1)
        per_class_context = self.rate_coded_context()
        self.context = torch.asarray(
            np.concatenate(per_class_context, axis=0), dtype=torch.float)
        
        
    def __getitem__(self, idx):
        return self.data[idx], self.context[idx], self.target[idx]
    
    def __len__(self):
        return len(self.data)

    

        
    