from tonic.datasets import SHD
import numpy as np
import h5py
import os
from tonic.io import make_structured_array
import torch

class SDHContext(SHD):
    """
    SHD gesture with context.
    Dataset consists of a sequence of seq_len dvs gesture sample from differents class and a one-hot coded context corresponding to a specific class.
    The task is to predict if a sample belong to the context observed.
    The target is a seq_len * sample_len vector which is constructed from two possible way define by target_type parameter:
    
    "memorization": Event memorization
        The task is to answer positively from the moment the sample of the given context is detected
    
        ex:
        Context (class representation): A
        Samples (class samples): b c a d e f
        Targets:                 F F T T T T
    
    "recognition": Local event dectection
        The Task is to answer positively only when the sample of the given context is detected:
        ex:
        Context (class representation): A
        Samples (class samples): b c a d e f
        Targets:                 F F T F F F
    
    In the memorization case, the class sample distribution is made such that the context class may not correspond to any sample in a given sequence.
    The expected prediction in that case would be to respond negativelly for all sample in the sequence.

    Args:
        save_to (str): dataset save path for the DVSgesture dataset
        seq_len (int): number of sample in a sequence
        cls_idx (int | list[int]): specific class to considers in the dataset, 
            if cls_idx is a integer we only consider the cls_idx fist classes 
        n_samples (int): number a sequence samples
        train (bool, optional): trainset or testset. Defaults to True.
        transform (Callable, optional): data transformation at sampling time. Defaults to None.
        context_transform (Callable, optional): context transformation at sampling time.. Defaults to None.
        target_transform (_type_, optional): target transformation at sampling time.. Defaults to None.
        task_type (str, optional): type of the task: [recognition, memory]. Defaults to "recognition".
    """
    def __init__(
        self,
        save_to,
        seq_len,
        c_classes,
        n_samples,
        train=True,
        transform=None,
        context_transform=None,
        target_transform=None,
        task_type: str = "recognition",
    ):
        super(SDHContext, self).__init__(
            save_to, train=train,
            transform=transform, target_transform=target_transform
            )
        
        # get mapping id -> target
        file = h5py.File(
            os.path.join(self.location_on_system, self.data_filename), "r")
  
        self.targets = [
            file["labels"][i].astype(int) for i in range(super().__len__())]
        self.data = np.arange(super().__len__())
        # nb_samples = super().__len__()
        
        # shuffle such that the train/valid cut is random
        # shuffle_idx = np.random.permutation(nb_samples)
        # self.data = [self.data[i] for i in shuffle_idx]
        # self.targets = [self.targets[i] for i in shuffle_idx]
        
        self.seq_len = seq_len
        self.c_classes = c_classes
        self.context_transform = context_transform
        self.n_samples = n_samples
        self.task_type = task_type.lower()
        
    def __getitem__(self, i):
        total_classes = set(self.c_classes)
        neg_class = np.random.choice(self.c_classes, (1,))
        pos_classes = list(total_classes - set(neg_class))
        pos_classes = np.random.choice(
            pos_classes, self.seq_len, replace=False)
        if self.task_type == "memory":
            prob = np.random.random()
            if prob < 0.5:
                context = np.random.choice(pos_classes, ())
            else:
                context = neg_class[0]
        else:
            # make such that each position is uniformly sampled
            if self.seq_len == 1:
                index = i % 2
                if index == 0:
                    context = neg_class[0]
                else:
                    context = pos_classes[0]
            else:
                target = i % (self.seq_len)
                context = pos_classes[target]

        events = None
        sequence = []
        targets = []
        block_idx = []
        current_block = 0
        sample_was_detected = False
        for c in pos_classes:
            idx = np.random.choice(np.flatnonzero(
                self.targets == c), size=())
            # follow the permutation to recovers the idx in file
            idx = self.data[idx]
            file = h5py.File(
                os.path.join(
                    self.location_on_system, self.data_filename), "r")
            events = make_structured_array(
                file["spikes/times"][idx] * 1e6,
                file["spikes/units"][idx],
                1,
                dtype=self.dtype,
            )
            if self.transform is not None:
                events = self.transform(events)
            sequence.append(events)
            
            sample_was_detected = (c == context) or (
                sample_was_detected and self.task_type == "memory")
            if sample_was_detected:
                current_target = 1
            else:
                current_target = 0
            block_idx.append(torch.full(
                size=(events.shape[0],), fill_value=current_block))
            current_block += 1
            targets.append(current_target)
        inp = torch.from_numpy(np.vstack(sequence))
        nb_timestep = inp.shape[0]

        # target = torch.from_numpy(np.array(target))
        target = torch.tensor(targets)
        block_idx = torch.cat(block_idx)
        
        if self.target_transform is not None:
            target = self.target_transform(target)

        context = torch.from_numpy(np.array(context))
        if self.context_transform is not None:
            context = self.context_transform(
                context, timesteps=nb_timestep)
        return inp, context, target, block_idx

    def __len__(self):
        return self.n_samples
