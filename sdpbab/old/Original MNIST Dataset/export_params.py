import os
import torch
from scipy.io import savemat
import pickle
import numpy as np
import gzip

from verinet.parsers.onnx_parser import ONNXParser
from calculate_bounds import BCalculator

# This script computes the pre-activation bounds for all layers in a feed-forward NN and stores them together with the
# weights and biases. The input bounds are derived from the Batten (2021) paper, the images from MNIST which are used
# are loaded from the original files

# Load MNIST pictures and labels
image_size = 28
num_images = 100
with gzip.open(os.path.join("data", "mnist", "train-images-idx3-ubyte.gz"), "r") as f_img:
    f_img.read(16)  # skip first four bytes (magic number, number of images, number of rows, number of columns)
    buf = f_img.read(image_size * image_size * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(num_images, image_size, image_size, 1)
    data = data / 255  # normalise

with gzip.open(os.path.join("data", "mnist", "train-labels-idx1-ubyte.gz"), "r") as f_lab:
    f_lab.read(8)  # skip first two bytes (magic number, number of items)
    buf = f_lab.read(num_images)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

# IMPORTANT: Change the epsilon according to the specifications in the Batten paper!
epsilon = 0.015  # For the epsilon, use the values from the Batten paper:
# 6x100: 0.026, 9x100: 0.026, 6x200: 0.015, 9x200: 0.015
dataset = "mnist"
nn_file = "mnist_relu_9_200.onnx"
path = os.path.join(dataset, nn_file)  # path to NN file
original_shape = [1, 28, 28]  # input shape for the network
flattened_shape = [original_shape[0] * original_shape[1] * original_shape[2]]
calculator = BCalculator(path, original_shape)

for image, (input_image, label) in enumerate(zip(data, labels)):
    lower_bounds = torch.flatten(torch.tensor(input_image)) - epsilon
    upper_bounds = torch.flatten(torch.tensor(input_image)) + epsilon
    inp_bounds = torch.stack((lower_bounds, upper_bounds), dim=-1)

    parser = ONNXParser(path)
    model = parser.to_pytorch()

    counter = 1
    savedict = dict()
    for idx, node in enumerate(model.nodes):
        if isinstance(node.op, torch.nn.modules.linear.Linear):
            # idx+1 because this calculates the pre-layer bounds and we want the bounds after the linear layer
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
    savedict["label"] = label  # save the label of the image that was used

    # save everything as a Matlab file
    print("Saving File for Image {}".format(image))
    savemat(os.path.join("data", "export", nn_file[:-5] + "_{}_image_{}.mat".format(epsilon, image)), savedict)
