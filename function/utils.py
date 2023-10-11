import torch
from typing import NamedTuple
import argparse

def merge_states(states: list[NamedTuple]) -> NamedTuple:
    """
    The layers have the possibility to record intermediary states for visualization. 
    
    The layer loop gives us a list of states, where each item in the list represents 
    the state of neurons at a specific time. 
    
    Our goal here is to combine these states into a single state where each field's 
    value becomes a new tensor with an added time dimension.

    
    Args:
        states (list[NamedTuple]): A list of states

    Returns:
        NamedTuple: A state with added temporal dimension for each fields value.
    """
    
    v = {k: torch.stack([getattr(ntuple, k) for ntuple in states], 0
                        ) for k in states[0]._asdict().keys()}
    return states[0].__class__(**v)

class StrToBool(argparse.Action):
    """Simple utility function to use shortcut for true-false in argparse.
    The default way to deal with boolean flag in argparse is cumbersome.

    """
    def __call__(self, parser, namespace, values, option_string=None):
        flag = values.lower()
        pos = ["true", "t", "1", "yes", "y"]
        if flag in pos:
            values = True
        else:
            values = False
        setattr(namespace, self.dest, values)