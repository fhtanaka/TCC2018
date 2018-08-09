import math
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from itertools import count
from PIL import Image
from memory import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def num_flat_features(x):
    size = x.size()[1:]  
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

def discover_flat_size(config):
    x = torch.zeros([config.batch_size, config.channels, config.height, config.width])
    relu = nn.ReLU()
    cont=0
    for in_features, out_features in zip(config.conv_layers, config.conv_layers[1:]):
        conv2d = nn.Conv2d (in_features, out_features, kernel_size=config.kernel[cont], padding=1)
        x = relu(conv2d(x))
        cont+=1

    return len(x[0][0][0])
    # return  len(x.view(-1, num_flat_features(x))[0])

class DQN(nn.Module):
    def __init__(self, config):

        super(DQN, self).__init__()

        #conv_layers = concolutional networks layer
        layer_count=0
        for in_features, out_features in zip(config.conv_layers, config.conv_layers[1:]):
            self.add_module("conv_layer"+str(layer_count), 
                nn.Sequential (
                    nn.Conv2d (in_features, out_features, kernel_size=config.kernel[layer_count], padding=1),
                    nn.ReLU()
                )
            )    
            layer_count+=1

        #nn_layers = neural networks layer
        
        #layer inicial que sai diretamente da conv
        self.add_module("nn_layer0", 
                nn.Sequential(
                    nn.Linear(discover_flat_size(config), config.nn_layers[0]),
                    nn.ReLU()
                )
            )
        if len(config.nn_layers) > 1:
            #layers intermediarias
            layer_count=1
            for in_features, out_features in zip(config.nn_layers, config.nn_layers[1:-1]):
                self.add_module("nn_layer"+str(layer_count), 
                    nn.Sequential(
                        nn.Linear(in_features, out_features),
                        nn.ReLU()
                    )
                )
                layer_count+=1
            #layer final, sem ReLu
            self.add_module("nn_layer"+str(layer_count), nn.Sequential(nn.Linear(config.nn_layers[-2], config.nn_layers[-1])))

    def forward(self, x):
        for layer in self._modules:
            print(layer)
            print(x.size())
            x = self._modules[layer](x)
        return x








