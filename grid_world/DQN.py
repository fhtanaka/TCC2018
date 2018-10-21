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
            m.weight.data.fill_(1.0)

class DQN(nn.Module):
    def __init__(self, config):

        super(DQN, self).__init__()

        #initial layer
        flat_features_size = (config.board_size+2*config.padding) - (config.kernel[0]-1)
        # self.add_module("conv_layer0",
        #         nn.Conv2d (config.channels, config.conv_layers[0], kernel_size=config.kernel[0], padding=0)
        # )

        if len(config.conv_layers) > 1:
            layer_count=1   
            for in_features, out_features in zip(config.conv_layers, config.conv_layers[1:]):
                self.add_module("conv_layer"+str(layer_count),
                        nn.Conv2d (in_features, out_features, kernel_size=config.kernel[layer_count], padding=0)
                )
                flat_features_size -= config.kernel[layer_count]-1
                layer_count+=1
        
        #full connected layer that connects the last conv layer to the Q-values
        out = (flat_features_size**2)*config.conv_layers[-1]
        self.add_module("full_connected_layer0", nn.Linear(out, 3))
        self.add_module("full_connected_layer1", nn.Linear(3, 4))

        for i in self.modules():
            i.apply(init_weights)

    def forward(self, x):
        layers_names = list(self._modules)
        x = x.view(-1, num_flat_features(x))
        for layer in layers_names[:-1]:
            x = F.relu(self._modules[layer](x))
            
        # This next line flat the vector tranforming the 3d tensor in a 1d
        # x = self._modules[layers_names[-2]](x)
        x = self._modules[layers_names[-1]](x)

        return x