import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import skimage

from interpret.chefer import *
from models.st_transformer import *
from models.pytorch_lightning import *
from data import *


def compute_average_representation(test_dataset, class_index):
    
    return None 

def compute_variance_representation(test_dataset, class_index):
    return None

# This way we can see how much of it overlaps 
def compute_average_masks(test_dataset, interpreter, class_index):
    return None 
