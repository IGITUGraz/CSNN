from dataclasses import dataclass
import torch
from tonic.transforms import ToFrame
import numpy as np
from torch.nn.functional import one_hot

from function.spike_encode import poisson_encode

@dataclass(frozen=True)
class BinarizeFrame(ToFrame):
    count_thr: float = 2
    def __call__(self, events):
        dense_frame: torch.sparse = super(
            BinarizeFrame, self).__call__(events)
        dense_frame = torch.from_numpy(dense_frame).float()
        dense_frame = torch.flatten(
            dense_frame, start_dim=-len(dense_frame.shape) + 1)
        dense_frame = torch.where(dense_frame >= self.count_thr, 1.0, 0.0)
        return dense_frame


@dataclass(frozen=True)
class TakeEventByTime:
    """ Take events in a certain time interval with a length proportional to 
    a specified ratio of the original length.

    Parameters:
        duration_interval (Union[float, Tuple[float]], optional):
        the length of the taken time interval, expressed in a ratio of the original sequence duration.
            - If a float, the value is used to calculate the interval length (0, duration_ratio)
            - If a tuple of 2 floats, the taken interval is [min_val, max_val]% 
            of the original sequence duration.
            Defaults to 0.2.

    Example:
        >>> transform = tonic.transforms.TakeEventByTime(duration_ratio=(0.1, 0.8))  # noqa
        (take the event part between 10% and 80%)
    """

    duration_interval: float | tuple[float] = 0.2

    def __call__(self, events):
        assert "x" and "t" and "p" in events.dtype.names
        # assert (
        #     type(self.duration_interval) == float and self.duration_interval >= 0.0 and self.duration_interval < 1.0
        # ) or (
        #     type(self.duration_interval) == tuple
        #     and len(self.duration_interval) == 2
        #     and all(val >= 0 and val < 1.0 for val in self.duration_interval)
        # )
        t_start = events["t"].min()
        
        t_end = events["t"].max()
        total_duration = (t_end - t_start)
        if isinstance(self.duration_interval, tuple):
            t_start = total_duration*self.duration_interval[0]
            t_end = total_duration*self.duration_interval[1]
        else:
            t_start = events["t"].min()
            t_end = total_duration*self.duration_interval
        mask_events = (events["t"] >= t_start) & (
            events["t"] <= t_end)
        mask_events = np.logical_not(mask_events)
        return np.delete(events, mask_events)

class PoissonContextTransform:
    def __init__(self,
                 c_classes: list[int],
                 f_min: float = 10.0,
                 f_max: float = 1000.0):
        self.c_classes = c_classes
        self.f_max = f_max
        self.f_min = f_min
        
    def __call__(self, ctx, timesteps):
        ctx = one_hot(ctx, num_classes=len(self.c_classes))
        spikes = poisson_encode(
            ctx, timesteps, f_max=self.f_max, f_min=self.f_min).squeeze(1)
        return spikes

@dataclass(frozen=True)
class Flatten:
    def __call__(self, data: np.ndarray):
        return data.reshape(data.shape[0], -1)