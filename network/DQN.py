import torch
import torch.nn as nn
import torch.nn.functional as F

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
                    nn.Conv2d (in_features, out_features, kernel_size=config.kernel[layer_count], padding=1)
            )    
            layer_count+=1

        #nn_layers = neural networks layer
        
        #layer inicial que sai diretamente da conv
        self.add_module("nn_layer0", nn.Linear(discover_flat_size(config), config.nn_layers[0]))
        if len(config.nn_layers) > 1:
            #layers intermediarias
            layer_count=1
            for in_features, out_features in zip(config.nn_layers, config.nn_layers[1:]):
                self.add_module("nn_layer"+str(layer_count), nn.Linear(in_features, out_features))
                layer_count+=1

    def forward(self, x):
        # print(self._modules)
        layers_names = list(self._modules)
        for layer in layers_names[:-1]:
            print(layer, '->', x.size())
            x = F.relu(self._modules[layer](x))
        x = self._modules[layers_names[-1]](x)
        return x








