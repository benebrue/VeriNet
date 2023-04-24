"""
Loads the .pth/.model files from the alpha-beta-CROWN toolbox and converts them to onnx models so that they can be used
with the VeriNet module.
Note: Some of these files are saved in a new PyTorch zip format and therefore require a newer version of PyTorch
Note 2: Using the automatic conversion to onnx provided by PyTorch produces files that break VeriNet, so a manual
conversion must be done (creating the VeriNetNN nodes step-by-step and setting their weights)
"""

import torch
import torch.nn as nn
import os
from verinet.neural_networks.verinet_nn import VeriNetNN, VeriNetNNNode


# mnist_cnn_a_adv.model, mnist_conv_small_nat.pth (alpha_beta_crown)

def mnist_cnn_4layer():
    # mnist_cnn_a
    return nn.Sequential(
        nn.Conv2d(1, 16, (4, 4), stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, (4, 4), stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(1568, 100),
        nn.ReLU(),
        nn.Linear(100, 10),
    )


def mnist_conv_small():
    return nn.Sequential(
        nn.Conv2d(1, 16, 4, stride=2, padding=0),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=0),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32*5*5, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )


def build_mnist_cnn_a(state_dict):
    nodes = []
    index = 0
    i = 0  # for iterating through the keys
    keys = list(state_dict.keys())

    nodes.append(VeriNetNNNode(index, torch.nn.Identity(), [], [index + 1]))
    index += 1

    new_op = torch.nn.Conv2d(1, 16, (4, 4), stride=2, padding=1)
    with torch.no_grad():
        new_op.weight = torch.nn.Parameter(state_dict[keys[i]])
        i += 1
        new_op.bias = torch.nn.Parameter(state_dict[keys[i]])
        i += 1
    nodes.append(VeriNetNNNode(index, new_op, [index - 1], [index + 1]))
    index += 1

    nodes.append(VeriNetNNNode(index, torch.nn.ReLU(), [index - 1], [index + 1]))
    index += 1

    new_op = torch.nn.Conv2d(16, 32, (4, 4), stride=2, padding=1)
    with torch.no_grad():
        new_op.weight = torch.nn.Parameter(state_dict[keys[i]])
        i += 1
        new_op.bias = torch.nn.Parameter(state_dict[keys[i]])
        i += 1
    nodes.append(VeriNetNNNode(index, new_op, [index - 1], [index + 1]))
    index += 1

    nodes.append(VeriNetNNNode(index, torch.nn.ReLU(), [index - 1], [index + 1]))
    index += 1

    nodes.append(VeriNetNNNode(index, torch.nn.Flatten(), [index - 1], [index + 1]))
    index += 1

    new_op = torch.nn.Linear(1568, 100)
    with torch.no_grad():
        new_op.weight = torch.nn.Parameter(state_dict[keys[i]])
        i += 1
        new_op.bias = torch.nn.Parameter(state_dict[keys[i]])
        i += 1
    nodes.append(VeriNetNNNode(index, new_op, [index - 1], [index + 1]))
    index += 1

    nodes.append(VeriNetNNNode(index, torch.nn.ReLU(), [index - 1], [index + 1]))
    index += 1

    new_op = torch.nn.Linear(100, 10)
    with torch.no_grad():
        new_op.weight = torch.nn.Parameter(state_dict[keys[i]])
        i += 1
        new_op.bias = torch.nn.Parameter(state_dict[keys[i]])
        i += 1
    nodes.append(VeriNetNNNode(index, new_op, [index - 1], [index + 1]))
    index += 1

    nodes.append(VeriNetNNNode(index, torch.nn.Identity(), [index - 1], []))

    model = VeriNetNN(nodes)

    return model


def build_mnist_conv_small(state_dict):
    nodes = []
    index = 0
    i = 0  # for iterating through the keys
    keys = list(state_dict.keys())

    nodes.append(VeriNetNNNode(index, torch.nn.Identity(), [], [index + 1]))
    index += 1

    new_op = torch.nn.Conv2d(1, 16, 4, stride=2, padding=0)
    with torch.no_grad():
        new_op.weight = torch.nn.Parameter(state_dict[keys[i]])
        i += 1
        new_op.bias = torch.nn.Parameter(state_dict[keys[i]])
        i += 1
    nodes.append(VeriNetNNNode(index, new_op, [index - 1], [index + 1]))
    index += 1

    nodes.append(VeriNetNNNode(index, torch.nn.ReLU(), [index - 1], [index + 1]))
    index += 1

    new_op = torch.nn.Conv2d(16, 32, 4, stride=2, padding=0)
    with torch.no_grad():
        new_op.weight = torch.nn.Parameter(state_dict[keys[i]])
        i += 1
        new_op.bias = torch.nn.Parameter(state_dict[keys[i]])
        i += 1
    nodes.append(VeriNetNNNode(index, new_op, [index - 1], [index + 1]))
    index += 1

    nodes.append(VeriNetNNNode(index, torch.nn.ReLU(), [index - 1], [index + 1]))
    index += 1

    nodes.append(VeriNetNNNode(index, torch.nn.Flatten(), [index - 1], [index + 1]))
    index += 1

    new_op = torch.nn.Linear(32*5*5, 100)
    with torch.no_grad():
        new_op.weight = torch.nn.Parameter(state_dict[keys[i]])
        i += 1
        new_op.bias = torch.nn.Parameter(state_dict[keys[i]])
        i += 1
    nodes.append(VeriNetNNNode(index, new_op, [index - 1], [index + 1]))
    index += 1

    nodes.append(VeriNetNNNode(index, torch.nn.ReLU(), [index - 1], [index + 1]))
    index += 1

    new_op = torch.nn.Linear(100, 10)
    with torch.no_grad():
        new_op.weight = torch.nn.Parameter(state_dict[keys[i]])
        i += 1
        new_op.bias = torch.nn.Parameter(state_dict[keys[i]])
        i += 1
    nodes.append(VeriNetNNNode(index, new_op, [index - 1], [index + 1]))
    index += 1

    nodes.append(VeriNetNNNode(index, torch.nn.Identity(), [index - 1], []))

    model = VeriNetNN(nodes)

    return model


folder = "alpha_beta_crown"
network = "mnist_conv_small_nat.pth"

# model = mnist_conv_small()  # change this to the correct model definition above
# model.load_state_dict(torch.load(os.path.join(folder, network)))  # for mnist_cnn_4layer
# model.load_state_dict(torch.load(os.path.join(folder, network))["state_dict"][0])  # for mnist_conv_small since the file
# "mnist_conv_small_nat.pth" contains a dictionary with the keys "acc" and "state_dict" and "state_dict" is a list which
# contains one element (the actual state_dict)
# model.eval()  # make sure model is in inference mode
# print(model)

# dummy_input = torch.randn(1, 1, 28, 28)
# input_names = ["input"]
# output_names = ["output"]
#
# torch.onnx.export(model, dummy_input, os.path.join(folder, network.split(".")[0] + ".onnx"), verbose=True,
#                   input_names=input_names, output_names=output_names)

if network == "mnist_conv_small_nat.pth":
    state_dict = torch.load(os.path.join(folder, network))["state_dict"][0]
    # the file "mnist_conv_small_nat.pth" contains a dictionary with the keys "acc" and "state_dict" and "state_dict"
    # is a list which contains one element (the actual state_dict)
else:
    state_dict = torch.load(os.path.join(folder, network))

model = build_mnist_conv_small(state_dict)

print(model)

# Save model to onnx
# model.save requires a dummy tensor with the same shape as the inputs to the network --> generated by torch.randn()
model.save(torch.randn(1, 1, 28, 28), os.path.join(folder, network.split(".")[0] + ".onnx"))
