import os
import torch
from scipy.io import savemat
import pickle

from verinet.parsers.onnx_parser import ONNXParser
from calculate_bounds import BCalculator

# This script computes the pre-activation bounds for all layers in a feed-forward NN and stores them together with the
# weights and biases. The input bounds are derived from the Batten (2021) paper, the images from MNIST which are used
# are loaded as numpy arrays from pickle files

path = os.path.join("mnist", "mnist_relu_4_1024.onnx")  # path to NN file
image = 9  # image from the MNIST dataset to be used
original_shape = [1, 28, 28]  # input shape for the network
epsilon = 0.03  # For the epsilon, use the values from the Batten paper: 6x100: 0.026, 6x200: 0.015
flattened_shape = [original_shape[0] * original_shape[1] * original_shape[2]]
calculator = BCalculator(path, original_shape)

with open(os.path.join("data", "save", "mnist_{}.pickle".format(image)), "rb") as input_file:
    input_image = pickle.load(input_file)  # load MNIST image from pickle file
lower_bounds = torch.flatten(torch.tensor(input_image)) - epsilon
upper_bounds = torch.flatten(torch.tensor(input_image)) + epsilon
inp_bounds = torch.stack((lower_bounds, upper_bounds), dim=-1)

parser = ONNXParser(path)
model = parser.to_pytorch()

counter = 1
savedict = dict()
for idx, node in enumerate(model.nodes):
    if isinstance(node.op, torch.nn.modules.linear.Linear):
        # idx+1 because this calculates the preactivation bounds and we want the bounds after the linear layer
        bounds = calculator.calculate_bounds(inp_bounds, layer=idx+1).numpy()
        weights = node.op.weight.detach().numpy()
        biases = node.op.bias.detach().numpy()

        # save data in a dict
        savedict["layer_{}_lb_pre".format(counter)] = bounds[:, 0]
        savedict["layer_{}_ub_pre".format(counter)] = bounds[:, 1]
        savedict["layer_{}_weights".format(counter)] = weights
        savedict["layer_{}_biases".format(counter)] = biases

        counter = counter + 1

savedict["input_image"] = input_image  # save the image that was used
print(savedict)

# save everything as a Matlab file
savemat(os.path.join("data", "export", "output_4_1024_epsilon_{}_image_{}.mat".format(epsilon, image)), savedict)
