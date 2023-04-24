"""
Loads an onnx model that was created by VeriNet and is therefore a VeriNetNNModel and converts it to a standard PyTorch
model before saving it as a .pth file which can be processed e.g. by alpha-beta-CROWN. Can only convert models that
consist of Linear, ReLU, Identity and Flatten layers, other layers have not been implemented yet.
"""

import torch
import os
import numpy as np
from verinet.neural_networks.verinet_nn import VeriNetNN, VeriNetNNNode
from verinet.parsers.onnx_parser import ONNXParser

folder = "small_mnist"  # small_mnist, self_trained_2_50_0.05, verigauge
net_name = "nips_pgd"  # nips_lp, nips_pgd, nips_sdp, self_trained_mnist, cifar10_A_certadv_2255

# Load model using VeriNet
path = os.path.join(folder, net_name + ".onnx")
parser = ONNXParser(path)
model = parser.to_pytorch()


# Convert to PyTorch model
class Network(torch.nn.Module):
    def __init__(self, verinet_model):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        with torch.no_grad():  # disable gradient computation
            for node in model.nodes:
                if isinstance(node.op, torch.nn.modules.linear.Identity):
                    # Do Nothing (Identity Layer) --> alpha-beta-CROWN can't process Identity layers, so we want to
                    # remove them from the model
                    # self.layers.append(torch.nn.Identity())
                    pass
                elif isinstance(node.op, torch.nn.modules.flatten.Flatten):
                    self.layers.append(torch.nn.Flatten())
                elif isinstance(node.op, torch.nn.modules.linear.Linear):
                    weights = node.op.weight
                    biases = node.op.bias
                    new_layer = torch.nn.Linear(weights.shape[1], weights.shape[0])
                    new_layer.weight = weights
                    new_layer.bias = biases
                    self.layers.append(new_layer)
                elif isinstance(node.op, torch.nn.modules.activation.ReLU):
                    self.layers.append(torch.nn.ReLU())
                else:
                    raise NotImplementedError("Processing for layer of type " + str(node.op) + " not implemented")
            self.layers.insert(0, torch.nn.Flatten())  # use this if Identity layers are not kept
            # self.layers.insert(1, torch.nn.Flatten())  # use this if Identity layers are kept

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


pytorch_model = Network(model)
print(pytorch_model.layers)
# Save model
path = os.path.join(folder, net_name + ".pth")

# remove "layers." prefix from keys in state_dict
state_dict = model.state_dict()
keys = list(state_dict.keys())
for key in keys:
    # new_key = str(int(key[7]) + 1) + key[8:]  # increment all key layer counts by 1 to match the keys generated
    # by the Sequential() model definitions. This is only needed if the Identity layers are kept, otherwise no
    # reindexing is required
    # when Identity layers are removed use this:
    new_key = key[7:]
    state_dict[new_key] = state_dict.pop(key)
print(state_dict.keys())

torch.save(state_dict, path)
