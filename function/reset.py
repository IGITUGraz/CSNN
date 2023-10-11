import torch 


def subtraction_reset(u_tm1: torch.Tensor, z_t: torch.Tensor, thr: float) -> torch.Tensor:
    return u_tm1 - z_t*thr


def zero_reset(u_tm1: torch.Tensor, z_t: torch.Tensor, thr: float) -> torch.Tensor:
    return u_tm1 * (1 - z_t)


def none_reset(u_tm1: torch.Tensor, z_t: torch.Tensor, thr: float) -> torch.Tensor:
    return u_tm1


reset_fn_map = {
    "subtraction": subtraction_reset,
    "zero": zero_reset,
    "none": none_reset, 
}

def get_reset_fn(method: str):
    """
    Exception catching function for clearer error message.
    """
    try:
        return reset_fn_map[method]
    except KeyError:
        print("Attempted to retrieve reset function {method}, but no such "
                    + "function exist. We currently support [subtraction, zero, none].")