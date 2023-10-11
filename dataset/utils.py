from copy import deepcopy
from dataclasses import dataclass
import io
import math
from pathlib import Path
import pickle
from typing import Any, Callable, Optional, Sequence
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import DataLoader

from tonic.dataset import Dataset as TonicDataset
import tonic

from dataset.transforms import BinarizeFrame, Flatten, PoissonContextTransform, TakeEventByTime
from .dvs import DVSGestureContext
from .shd import SDHContext


class PadTensors:
    """This is a custom collate function for a pytorch dataloader to load multiple
    event recordings at once.
    """

    def __init__(self, batch_first: bool = True):
        self.batch_first = batch_first
        self.batch_dim = 0 if batch_first else 1
        self.temp_dim = 1 if batch_first else 0
        self.batch_first
        
    def __call__(self,batch: list[tuple[torch.Tensor, torch.Tensor, int]]
        ) -> tuple[tuple[torch.Tensor, torch.Tensor,], 
                       tuple[torch.Tensor, torch.Tensor]]:
        sample: torch.Tensor
        context: torch.Tensor
        target: torch.Tensor
        transposed_batch = list(zip(*batch))  # type: ignore
        sample = transposed_batch[0]
        context = transposed_batch[1]
        target_list = transposed_batch[2]
        target_block_idx = transposed_batch[3]
        
        target = torch.stack(target_list, 0)
        
        # get the number of class + 1
        padding_number = target.size(1)
        sample = pad_sequence(sample, batch_first=self.batch_first)

        target_block_idx = pad_sequence(
            target_block_idx, batch_first=self.batch_first, 
            padding_value=padding_number).long()
        context = pad_sequence(context, batch_first=self.batch_first)
        return sample, context, target, target_block_idx

def split_dataset(dataset: TonicDataset, percent: float, shuffle_before_split=True):
    if shuffle_before_split:
        perm = np.random.permutation(len(dataset.data))
        dataset.data = np.array(dataset.data)[perm]
        dataset.targets = np.array(dataset.targets)[perm]
        
    train_dataset = dataset
    val_dataset = deepcopy(dataset)
    train_data = [dataset.data[i] for i in range(
        0, math.floor(percent*len(dataset.data)))]
    train_targets = [dataset.targets[i] for i in range(
        0, math.floor(percent*len(dataset.data)))]

    val_data = [dataset.data[i] for i in range(
        math.floor(percent*len(dataset.data)), len(dataset.data))]
    val_targets = [dataset.targets[i] for i in range(
        math.floor(percent*len(dataset.data)), len(dataset.data))]
    train_dataset.data = train_data
    train_dataset.targets = train_targets
    val_dataset.data = val_data
    val_dataset.targets = val_targets
    val_dataset.n_samples = int(
        math.floor((1 - percent)*train_dataset.n_samples))
    return train_dataset, val_dataset


@dataclass
class DiskCachedDataset(torch.utils.data.Dataset):
    dataset: Sequence
    contextual: bool
    cache_path: str | Path
    reset_cache: bool = False
    num_copies: int = 1
    transform: Optional[Callable] = None
    target_transform: Optional[Callable] = None
    context_transform: Optional[Callable] = None
    
    def __post_init__(self):
        super().__init__()
        self.cache_path = Path(self.cache_path)
        self.cache_path.mkdir(exist_ok=True, parents=True)
        if self.dataset is None:
            filenames = [
                name for name in self.cache_path.iterdir() if name.is_file()
                ]
            self.n_samples = (len(filenames) // self.num_copies)
        else:
            self.n_samples = len(self.dataset)
        
    def io(self, key, item=None):
        data = None
        if item is None:
            item = key
        try:
            f = open(self.cache_path / key, 'rb')
            buffer = io.BytesIO(f.read())
            data = torch.load(buffer)
        except FileNotFoundError:
            data = self.dataset[item]
            torch.save(data, self.cache_path / key, pickle_protocol=pickle.HIGHEST_PROTOCOL)
        return data
    

    def __getitem__(self, item
                    ) -> tuple[object, object, object] | tuple[object, object]:
        copy = np.random.randint(self.num_copies)
        key = f"{item}_{copy}"
        # format might change during save to hdf5,
        # i.e. tensors -> np arrays
        raw_data = self.io(key, item)
        if self.contextual:
            data, context, targets, block_idx = raw_data
        else:
            data, targets, block_idx = raw_data
        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            targets = self.target_transform(targets)
        if self.contextual:
            if self.context_transform is not None:
                context = self.context_transform(context, 
                                                 timesteps=data.size(0))
            return data, context, targets, block_idx
        else:
            return data, targets, block_idx

    def __len__(self):
        return self.n_samples

def setup_dataloader(dataset_type: str, 
                     save_to: str,seq_len: int, 
                     c_classes: int| list[int],
                     n_samples: int,
                     task_type: str,
                     spatial_factor: float,
                     time_factor: float,
                     time_window: float,
                     count_thr: int,
                     duration_ratio: float,
                     ctx_f_max: float,
                     split_percent: float,
                     batch_size: int,
                     dataloader_nb_workers: int = 1,
                     device: Optional[torch.device] = None,
                     ):
    
    # select a specific data 
    # create the dataset at a given location
    # split on multiple splits
    # create required transform for the data and context
    # create the dataloader
    if dataset_type.lower() == "dvsgesture":
        dataset_class = DVSGestureContext
    elif dataset_type.lower() == "shd":
        dataset_class = SDHContext
    else:
        raise ModuleNotFoundError(f"dataset {dataset_type.lower()} doesn't exist."\
                                  "Available dataset are [dvsgesture, shd]")
    sensor_size = dataset_class.sensor_size
    print(sensor_size)
    sensor_size = (
        int(math.ceil(sensor_size[0] * spatial_factor)),
        int(math.ceil(sensor_size[1] * spatial_factor)),
        sensor_size[2]
        )
    
    input_transform = tonic.transforms.Compose([
        TakeEventByTime(duration_ratio),
        tonic.transforms.Downsample(
            time_factor=time_factor,
            spatial_factor=spatial_factor
        ),
        BinarizeFrame(
            sensor_size=sensor_size,
            time_window=time_window,
            count_thr=count_thr),
        Flatten(),
    ])
    if isinstance(c_classes, int):
        c_classes = list(range(c_classes))
        
    context_transform = PoissonContextTransform(c_classes=c_classes, f_max=ctx_f_max)

    train_dataset = dataset_class(
        save_to=save_to,
        seq_len=seq_len,
        c_classes=c_classes,
        n_samples=n_samples,
        train=True,
        task_type=task_type,
        transform=input_transform,
        context_transform=context_transform
    )
    test_dataset = dataset_class(
        save_to=save_to,
        seq_len=seq_len,
        c_classes=c_classes,
        n_samples=n_samples,
        train=False,
        task_type=task_type,
        transform=input_transform,
        context_transform=context_transform
    )
    input_size = math.prod(sensor_size)
    
    # class_weighting for unbalanced task
    if task_type == "recognition":
        class_weights = torch.full(
             size=(), fill_value=max(1, seq_len - 1))
    else:
        weight = (3*seq_len - 1)/(seq_len + 1)
        class_weights = torch.full(
            (), fill_value=weight)
    
    cache_path = f"{save_to}/cache/{dataset_type.lower()}/" \
        f'{seq_len}_{"-".join(str(x) for x in train_dataset.c_classes)}' \
        f'_{spatial_factor}_{time_factor}'\
        f'_{time_window}_{count_thr}'\
        f'_{duration_ratio}_{task_type}_{ctx_f_max}'
    
    train_dataset, val_dataset = split_dataset(
        train_dataset, split_percent, shuffle_before_split=True)
    
    collate_fn = PadTensors(batch_first=True)
    train_dataset = DiskCachedDataset(
        train_dataset, contextual=True,
        cache_path=cache_path + "/train",
    )
    val_dataset = DiskCachedDataset(
        val_dataset, contextual=True,
        cache_path=cache_path + "/val",
    )
    test_dataset = DiskCachedDataset(
        test_dataset, contextual=True,
        cache_path=cache_path + "/test",
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
        num_workers=dataloader_nb_workers,
        persistent_workers=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
        num_workers=dataloader_nb_workers,
        persistent_workers=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
        num_workers=dataloader_nb_workers,
        persistent_workers=True,
    )
    return train_dataloader, val_dataloader, test_dataloader, input_size, class_weights