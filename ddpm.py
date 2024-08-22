import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from functions.ckpt_utils import download_cifar10_checkpoint
import logging
from torch.utils.tensorboard import SummaryWriter

# TODO think about unet model to use
