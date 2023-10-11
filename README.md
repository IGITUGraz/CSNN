# Context-Dependent Computations in Spiking Neural Networks with Apical Modulation

## Abstract

This paper explores the integration of bottom-up and top-down information in neocortical pyramidal neurons, highlighting its importance for cortical computations. We propose a simple spiking neuron model that captures the properties of top-down modulated activity. To investigate the impact of this model on context-dependent computations, we evaluate recurrently connected networks of such neurons. Our results demonstrate the enhanced capability of spike-based context-dependent computations.

### Keywords

Spiking Neural Networks, Pyramidal Neurons, Apical Modulation, Context-Dependent Computations

## How to Install and Run the Program

### Required Pip Package

To run the simulation program used in this research, you will need to install the following Python package:
- pytorch
- torchmetrics
- tonic


### Run 
```bash
$ python run.py --help
usage: CSNN [-h] [--seed SEED] [--nb_worker NB_WORKER] [--data_path DATA_PATH] [--batch_size BATCH_SIZE] [--dataset {dvsgesture,shd}] [--seq_len SEQ_LEN]
                                  [--task_type {recognition,memory}] [--model_type {clif,add_clif,lif,lsnn}] [--apical_rec {true,t,1,yes,y,false,f,0,no,n}]
                                  [--apical_tau {short,long,distributed}] [--lr LR] [--lr_decay LR_DECAY] [--l2_decay L2_DECAY] [--epoch EPOCH] [--target_spike_prob TARGET_SPIKE_PROB]
                                  [--spike_reg_l2 SPIKE_REG_L2]

options:
  -h, --help            show this help message and exit
  --seed SEED           Random seed for PyTorch and Numpy.
  --nb_worker NB_WORKER
                        Number of workers for the pytorch dataloaders.
  --data_path DATA_PATH
                        Path for dataset download and caching location.
  --batch_size BATCH_SIZE
                        Batch size
  --dataset {dvsgesture,shd}
                        Dataset choice.
  --seq_len SEQ_LEN     Number of item in the sequence, in the paper we evaluate for 1 and 5.
  --task_type {recognition,memory}
                        Select either the memorization of recognition task.
  --model_type {clif,add_clif,lif,lsnn}
                        Model choice.
  --apical_rec {true,t,1,yes,y,false,f,0,no,n}
                        Determine if we want to use apical recurrence for clif models.
  --apical_tau {short,long,distributed}
                        Determine the time scale of the apical membrane time constant short = 20ms, long = 200ms, distributed unif(20, 200)ms
  --lr LR               Learning step.
  --lr_decay LR_DECAY   Exponential decay rate of the learning rate.
  --l2_decay L2_DECAY   Weights decay factor.
  --epoch EPOCH         Number of epoch.
  --target_spike_prob TARGET_SPIKE_PROB
                        Target spike probability for regularization.
  --spike_reg_l2 SPIKE_REG_L2
                        Target spike probability for regularization.
```