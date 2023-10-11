import sys, os, math, copy, random, time, logging
from argparse import ArgumentParser
import argparse
from typing import Any, Optional
import torch
import numpy as np
from dataset.utils import setup_dataloader
from torch.utils.data import DataLoader
from function.regularization import layer_wise_regularization
from module import CLIFLayer, ALIFLayer, LIFLayer, LILayer, SensoryContextConcat, CustomSeq
from function.utils import StrToBool
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall

def setup_dataset_and_model(config: argparse.Namespace):
    # namespace to dict
    config: dict = vars(config)
    seed = config["seed"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(device)
    # Dataset
    thr = 0.05
    if config["dataset"] == "dvsgesture":
        c_classes = 10
        nb_samples = 2048
        spatial_factor = 0.12
        time_factor = 1e-3
        time_window = 5.0
        count_thr = 3
        duration_ratio = 0.4
        rate_soma = 43
        rate_apical = 20
        
    elif config["dataset"] == "shd":
        c_classes = 20
        nb_samples = 4096
        spatial_factor = 0.5
        time_factor = 1e-3
        time_window = 2
        count_thr = 1
        duration_ratio = 1.0
        rate_soma = 50
        rate_apical = 20
        
    train_dl, val_dl, test_dl, input_size, class_weight = setup_dataloader(
        config["dataset"], config["data_path"], 
        config["seq_len"], c_classes, nb_samples, 
        config["task_type"], spatial_factor, time_factor, time_window, count_thr,
        duration_ratio, ctx_f_max=200, split_percent=0.8, batch_size=config["batch_size"],
        dataloader_nb_workers=config["nb_worker"], device=device)
    
    layers = []
    map_tau = {
        "short": 20 if config["dataset"] == "dvsgesture" else 40,
        "long": 200,
        "distributed": (20, 200)
    }
    if config["model_type"] not in ["clif", "add_clif"]:
        layers.append(SensoryContextConcat())
    if config["model_type"] == "lif":
        l1 = LIFLayer(input_size + c_classes, 200, 20, thr,
                      dt=1)
    elif config["model_type"] == "lsnn":
        l1 = ALIFLayer(input_size + c_classes, 200, tau_soma_range=20,
                       tau_beta_range=map_tau[config["apical_tau"]],
                       input_rate=rate_soma+rate_apical,
                       threshold=thr, thr_incr=1.3)
    elif config["model_type"] in ["clif", "add_clif"]:
        map_interaction = {
            "clif": "relu_mult",
            "add_clif": "add"
        }
        l1 = CLIFLayer(input_size, c_classes, 200, threshold=thr,
                       tau_soma_range=20, tau_apical_range=map_tau[config["apical_tau"]],
                       use_apical_recurrent=config["apical_rec"],
                       interaction_type=map_interaction[config["model_type"]],
                       rate_soma=rate_soma, rate_apical=rate_apical)
    layers.append(l1)
    l2 = LILayer(200, 1, input_rate=1)
    layers.append(l2)
    model = CustomSeq(*layers).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr = config["lr"], 
                                 weight_decay=config["l2_decay"])
    
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config["lr_decay"])
    
    loss_fn = BCEWithLogitsLoss(
        reduction='mean',
        pos_weight=class_weight)
        # reg_loss = self.l2*regularization_multi_layers(
        #     states, 1e-3 * inputs[0].size(1), self.target_rate, 0, "upper")
        
    return (train_dl, val_dl, test_dl), model, loss_fn, optimizer, lr_scheduler, config, device

def by_block_summation(outputs, targets, block_idx):
    block_idx = block_idx.unsqueeze(2).expand(
        size=(-1, -1, outputs.size(2)))
    block_outputs = torch.zeros(
        size=(
            targets.shape[0],
            targets.size(1) + 1,
            outputs.size(2)
            ), 
        dtype=outputs.dtype,
        device=outputs.device)
    # outputs = torch.tanh(outputs)
    block_outputs.scatter_reduce_(
        1, index=block_idx, src=outputs, reduce="mean")
    if block_outputs.shape[1] > targets.shape[1]:
        block_outputs = block_outputs[:, :-1]
    outputs_reduce = block_outputs.squeeze()
    targets_reduce = targets.squeeze()
    return outputs_reduce, targets_reduce

def fit_and_test(
    config: dict[str, Any], optimizer: torch.nn.Module, 
    lr_scheduler: torch.nn.Module, loss_fn: torch.nn.Module, 
    model: torch.nn.Module, train_dataloader: DataLoader, 
    val_dataloader: Optional[DataLoader] = None, 
    test_dataloader: Optional[DataLoader]= None,
    device: Optional[torch.device] = None):
    acc = BinaryAccuracy().to(device)
    precision = BinaryPrecision().to(device)
    recall = BinaryRecall().to(device)
    # in the memorization task we train on the full sequence, but evaluate on the last item
    # this is because we want to evaluate the memorization capabilities.
    # the last item have equiprobable pos/neg target probability.
    eval_at_n = -1
    if config["task_type"] == "memory":
        eval_at_n = config["seq_len"] - 1
    for i in range(config["epoch"]):
        epoch_loss = 0
        for batch in tqdm(train_dataloader, "Training epoch"):
            optimizer.zero_grad()
            x, c, targets, block_idx = batch


            x = x.to(device)
            c = c.to(device)
            targets = targets.to(device)
            block_idx = block_idx.to(device)
            time_steps = (block_idx < config["seq_len"]).sum(1)
            outputs = model((x, c))
            spike_sum = model.get_spike_sums()
            
            reg_loss, spike_proba = layer_wise_regularization(spike_sum[0], config["target_spike_prob"], time_steps)
            outputs_reduce, targets_reduce = by_block_summation(
                outputs, targets, block_idx)
            acc(outputs_reduce, targets_reduce)
            precision(outputs_reduce, targets_reduce)
            recall(outputs_reduce, targets_reduce)
            loss = loss_fn(outputs_reduce, targets_reduce.float())
            loss += config["spike_reg_l2"] * reg_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss
        lr_scheduler.step()
        train_epoch_loss =  epoch_loss / len(train_dataloader) 
        train_acc = acc.compute()
        train_precision = precision.compute()
        train_recall = recall.compute()
        acc.reset(), precision.reset(), recall.reset()
        epoch_loss = 0
        if val_dataloader is not None:
            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc="Validation epoch:"):
                    x, c, targets, block_idx = batch
                    x = x.to(device)
                    c = c.to(device)
                    targets = targets.to(device)
                    block_idx = block_idx.to(device)
                    outputs = model((x, c))
                    outputs_reduce, targets_reduce = by_block_summation(
                        outputs, targets, block_idx)
                    if eval_at_n != -1:
                        outputs_reduce = outputs_reduce[:, eval_at_n]
                        targets_reduce = targets_reduce[:, eval_at_n]
                    acc(outputs_reduce, targets_reduce)
                    precision(outputs_reduce, targets_reduce)
                    recall(outputs_reduce, targets_reduce)
                    loss = loss_fn(outputs_reduce, targets_reduce.float())
                    epoch_loss += loss
            with torch.no_grad():
                val_epoch_loss =  epoch_loss / len(val_dataloader) 
                val_acc = acc.compute()
                val_precision = precision.compute()
                val_recall = recall.compute()
                str_format = "Epoch n-{} train_loss: {}, train_acc: {}, train_precision: {}, train_recall {}\n"\
                            "           vall_loss: {}, val_acc: {}, val_precision: {}, val_recall {}"
                    
                print(str_format.format(i+1, train_epoch_loss, train_acc, train_precision, train_recall,
                                        val_epoch_loss, val_acc, val_precision, val_recall))
                acc.reset(), precision.reset(), recall.reset()
                epoch_loss = 0
    if test_dataloader is not None:
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Testing epoch:"):
                x, c, targets, block_idx = batch
                x = x.to(device)
                c = c.to(device)
                targets = targets.to(device)
                block_idx = block_idx.to(device)
                outputs = model((x, c))
                outputs_reduce, targets_reduce = by_block_summation(
                    outputs, targets, block_idx)
                if eval_at_n != -1:
                    outputs_reduce = outputs_reduce[:, eval_at_n]
                    targets_reduce = targets_reduce[:, eval_at_n]
                acc(outputs_reduce, targets_reduce)
                precision(outputs_reduce, targets_reduce)
                recall(outputs_reduce, targets_reduce)
                loss = loss_fn(outputs_reduce, targets_reduce.float())
                epoch_loss += loss
            test_epoch_loss =  epoch_loss / len(test_dataloader) 
            test_acc = acc.compute()
            test_precision = precision.compute()
            test_recall = recall.compute()
        str_format = "Testing: loss: {}, accuracy: {}, precision: {}, recall {}"
        print(str_format.format(test_epoch_loss, test_acc, test_precision, test_recall))
if __name__ == '__main__':
    parser = ArgumentParser("CSNN")
    parser.add_argument(
        "--seed",
        type=int,
        default=123123,
        help="Random seed for PyTorch and Numpy."
    )
    parser.add_argument(
        "--nb_worker",
        type=int,
        default=1,
        help="Number of workers for the pytorch dataloaders."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data",
        help="Path for dataset download and caching location."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dvsgesture",
        choices=["dvsgesture", "shd"],
        help="Dataset choice."
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=1,
        help="Number of item in the sequence, in the paper we evaluate for 1 and 5."
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default="recognition",
        choices=["recognition", "memory"],
        help="Select either the memorization of recognition task."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="clif",
        choices=["clif","add_clif", "lif", "lsnn"],
        help="Model choice."
    )
    parser.add_argument(
        "--apical_rec",
        action=StrToBool,  # noqa
        default=True,
        choices=["true", "t", "1", "yes", "y", "false", "f", "0", "no", "n"],
        help=f"Determine if we want to use apical recurrence for clif models."
    )
    parser.add_argument(
        "--apical_tau",
        type=str,
        default="short",
        choices=["short", "long", "distributed"],
        help="Determine the time scale of the apical membrane time constant short = 20ms, long = 200ms, distributed unif(20, 200)ms"
    )
    parser.add_argument(
        "--lr",
        default=1e-2,
        type=float,
        help="Learning step."
    )
    parser.add_argument(
        "--lr_decay",
        default=0.85,
        type=float,
        help="Exponential decay rate of the learning rate."
    )
    parser.add_argument(
        "--l2_decay",
        default=0.0,
        type=float,
        help="Weights decay factor."
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=1,
        help="Number of epoch."
    )
    parser.add_argument(
        "--target_spike_prob",
        type=float,
        default=0.01,
        help="Target spike probability for regularization."
    )
    parser.add_argument(
        "--spike_reg_l2",
        type=float,
        default=0.0001,
        help="Target spike probability for regularization."
    )
    dataloaders, model, loss_fn, optimizer, lr_scheduler, config, device = setup_dataset_and_model(parser.parse_args())
    train_dl, val_dl, test_dl = dataloaders
    fit_and_test(config, optimizer, lr_scheduler, loss_fn, model, train_dl, val_dl, test_dl, device)
