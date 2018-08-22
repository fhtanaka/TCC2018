import torch
import torch.nn as nn
import torch.nn.functional as F

def num_flat_features(x):
    size = x.size()[1:]  
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

def discover_flat_size(config):
    x = torch.zeros([config.channels, config.height, config.width])
    cont=1
    conv2d = nn.Conv2d(config.channels, config.conv_layers[0], kernel_size=config.kernel[0], padding=0)
    x = F.relu(conv2d(x))
    for in_features, out_features in zip(config.conv_layers, config.conv_layers[1:]):
        conv2d = nn.Conv2d (in_features, out_features, kernel_size=config.kernel[cont], padding=0)
        x = F.relu(conv2d(x))
        cont+=1

    return len(x.view(-1, num_flat_features(x))[0])
    # return  len(x.view(-1, num_flat_features(x))[0])

class DQN(nn.Module):
    def __init__(self, config):

        super(DQN, self).__init__()

        #layer inicial
        flat_features_size = config.board_size - (config.kernel[0]-1)
        self.add_module("conv_layer0",
            nn.Sequential(
                nn.Conv2d (config.channels, config.conv_layers[0], kernel_size=config.kernel[0], padding=0),
                nn.BatchNorm2d(config.conv_layers[0])
            ) 
        )
        if len(config.conv_layers) > 1:
            layer_count=1   
            for in_features, out_features in zip(config.conv_layers, config.conv_layers[1:]):
                self.add_module("conv_layer"+str(layer_count),
                    nn.Sequential(
                        nn.Conv2d (in_features, out_features, kernel_size=config.kernel[layer_count], padding=0),
                        nn.BatchNorm2d(out_features)    
                    )
                )
                flat_features_size -= config.kernel[layer_count]-1
                layer_count+=1
        
        #layer inicial que sai diretamente da conv
        self.add_module("nn_layer0", nn.Linear((flat_features_size**2)*config.conv_layers[-1], config.board_size**2))
        # if len(config.nn_layers) > 1:
        #     #layers intermediarias
        #     layer_count=1
        #     for in_features, out_features in zip(config.nn_layers, config.nn_layers[1:]):
        #         self.add_module("nn_layer"+str(layer_count), nn.Linear(in_features, out_features))
        #         layer_count+=1
        # self.add_module("final_layer", nn.Linear(config.nn_layers[-1], config.board_size**2))

    def forward(self, x):
        # print(self._modules)
        layers_names = list(self._modules)
        for layer in layers_names[:-1]:
            x = F.relu(self._modules[layer](x))
        
        x = x.view(-1, num_flat_features(x))
        x = self._modules[layers_names[-1]](x)

        return x