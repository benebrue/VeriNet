"""
Loads the .pth files from the VeriGauge toolbox and converts them to onnx models so they can be used with the VeriNet
module. Makes strong assumptions on the network structure: only Linear and ReLU layers, no ReLU after last Linear layer
"""

import torch
import os
import numpy as np
from verinet.neural_networks.verinet_nn import VeriNetNN, VeriNetNNNode

folder = "verigauge"  # one of verigauge

net_name = "mnist_cnn_a_adv.model"
# one of cifar10_A_certadv_2255.pth, cifar10_B_certadv_2255.pth, mnist_G_certadv_1.pth, mnist_G_certadv_2.pth
# (VeriGauge)

path = os.path.join(folder, net_name)

net_dict = torch.load(path, map_location=torch.device("cpu"))
state_dict = net_dict["state_dict"]
keys = list(net_dict["state_dict"].keys())  # keys of the ordered dict, necessary since odict isn't iterable

nodes = []
index = 0
nodes.append(VeriNetNNNode(index, torch.nn.Identity(), [], [index+1]))
index += 1
for i in range(0, len(state_dict), 2):
    new_op = torch.nn.Linear(state_dict[keys[i]].shape[1], state_dict[keys[i]].shape[0])
    with torch.no_grad():
        new_op.weight = torch.nn.Parameter(state_dict[keys[i]])
        new_op.bias = torch.nn.Parameter(state_dict[keys[i+1]])
    nodes.append(VeriNetNNNode(index, new_op, [index-1], [index+1]))
    index += 1
    if i != (len(net_dict["state_dict"]) - 2):  # not the last node, -2 because it starts at 0 and counts in steps of 2
        nodes.append(VeriNetNNNode(index, torch.nn.ReLU(), [index-1], [index+1]))
        index += 1

nodes.append(VeriNetNNNode(index, torch.nn.Identity(), [index-1], []))

model = VeriNetNN(nodes)
print(model)

# Save model to onnx
# model.save requires a dummy tensor with the same shape as the inputs to the network --> generated by torch.randn()
model.save(torch.randn(1, state_dict[keys[0]].shape[1]), os.path.join("verigauge", net_name + ".onnx"))
