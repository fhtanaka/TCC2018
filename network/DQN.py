import torch
import torch.nn as nn
import torch.nn.functional as F

def num_flat_features(x):
    size = x.size()[1:]  
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

def init_weights(m):
    if type(m) == nn.Linear:
        m.weight.data.fill_(.0)

class DQN(nn.Module):
    def __init__(self, config):

        super(DQN, self).__init__()


        flat_features_size = (config.board_size+2*config.padding)

        # Conv layers
        conv_layers = [config.channels, *config.conv_layers]
        for i in range(0, len(conv_layers)-1):
            self.add_module("conv_layer" + str(i),
                nn.Conv2d(conv_layers[i], conv_layers[i+1], kernel_size=config.kernel[0], padding=0)
            )
            flat_features_size -= config.kernel[i]-1
        
        if (len(conv_layers) > 1):
            out = (flat_features_size**2)*config.conv_layers[-1]
            self.flatten = lambda x: x.view(-1, num_flat_features(x))
        else:
            out = (flat_features_size**2)*config.channels
            self.flatten = lambda x: x.view(-1, num_flat_features(F.relu(x)))

        #full connected layer that connects the last conv layer to the Q-values
        self.add_module("full_connected_layer", nn.Linear(out, config.board_size**2))
        self._modules["full_connected_layer"].apply(init_weights)

    def forward(self, x):
        layers_names = list(self._modules)
        for layer in layers_names[:-1]:
            x = F.relu(self._modules[layer](x))
            
        # This next line flat the vector tranforming the 3d tensor in a 1d
        x = self.flatten(x)
        x = self._modules[layers_names[-1]](x)

        return x