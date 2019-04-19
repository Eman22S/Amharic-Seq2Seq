#!/usr/bin/env python
# coding: utf-8
import yaml
import torch
import random
import argparse
import numpy as np
from src.solver import Tester as Solver

# Make cudnn CTC deterministic
torch.backends.cudnn.deterministic = True

# Arguments


parser = argparse.ArgumentParser(description='Training E2E asr.')
parser.add_argument('--config', type=str, default="config/alffa_example.yaml", help='Path to experiment config.')
parser.add_argument('--name', default=None, type=str, help='Name for logging.')
parser.add_argument('--logdir', default='log/', type=str, help='Logging path.', required=False)
parser.add_argument('--ckpdir', default='result/', type=str, help='Checkpoint/Result path.', required=False)
parser.add_argument('--load', default=None, type=str, help='Load pre-trained model', required=False)
parser.add_argument('--seed', default=0, type=int, help='Random seed for reproducable results.', required=False)
parser.add_argument('--njobs', default=8, type=int, help='Number of threads for decoding.', required=False)
parser.add_argument('--cpu', action='store_true', help='Disable GPU training.')
parser.add_argument('--test', action='store_true', help='Test the model.')
parser.add_argument('--no-msg', action='store_true', help='Hide all messages.')
parser.add_argument('--rnnlm', action='store_true', help='Option for training RNNLM.')
paras = parser.parse_args()
setattr(paras,'gpu',not paras.cpu)
setattr(paras,'verbose',not paras.no_msg)
config = yaml.load(open(paras.config,'r'))
random.seed(paras.seed)
np.random.seed(paras.seed)
torch.manual_seed(paras.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(paras.seed)
solver = Solver(config,paras)
solver.set_model()

def transcript(filename):
    result = solver.exec_file(filename)
    return result

res = transcript('01_d501021.npy')
print(res)
