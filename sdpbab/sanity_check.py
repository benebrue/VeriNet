import os
import torch

from verinet.parsers.onnx_parser import ONNXParser
from calculate_bounds import BCalculator
from calculate_naive_bounds import NaiveBCalculator

path = os.path.join("mnist", "mnist_relu_3_50.onnx")
original_shape = [1, 28, 28]  # input shape for the MNIST dataset
# path = os.path.join("cifar10", "ffnnRELU__Point_6_500.onnx")
# original_shape = [3, 32, 32]  # input shape for the MNIST dataset
# layer = 7  # watch out: RSIP returns bounds before layer l while naive calculator returns bounds after layer l
flattened_shape = [original_shape[0] * original_shape[1] * original_shape[2]]

for layer in range(1, 8):
    input_image = torch.full(flattened_shape, 0.5)
    lower_bounds_flat = input_image - 0.01
    upper_bounds_flat = input_image + 0.01
    inp_bounds_flat = torch.stack((lower_bounds_flat, upper_bounds_flat), dim=-1)
    parser = ONNXParser(path)
    model = parser.to_pytorch()
    calculator = BCalculator(model, original_shape)
    bounds = calculator.calculate_bounds(inp_bounds_flat, layer=layer+1).numpy()

    input_image = torch.full(original_shape, 0.5)
    lower_bounds = input_image - 0.01
    upper_bounds = input_image + 0.01
    inp_bounds = torch.stack((lower_bounds, upper_bounds), dim=-1)
    ncalculator = NaiveBCalculator(path, original_shape)
    naive_bounds = ncalculator.calculate_naive_bounds(inp_bounds, layer=layer).numpy()

    # Compare lower bounds
    tighter = naive_bounds[:, 0] < bounds[:, 0]
    proportion_lower = int((tighter.sum() / bounds.shape[0])*100)

    # Compare upper bounds
    tighter = bounds[:, 1] < naive_bounds[:, 1]
    proportion_upper = int((tighter.sum() / bounds.shape[0])*100)

    print("Layer {}: Lower Bounds: {}%, Upper Bounds: {}%".format(layer, proportion_lower, proportion_upper))
    print(bounds)
    print(naive_bounds)

# print("Bound Comparison - Lower Bounds:")
# print(naive_bounds[:, 0] < bounds[:, 0])
#
# print("Bound Comparison - Upper Bounds:")
# print(bounds[:, 1] < naive_bounds[:, 1])
#
# print(bounds[-10:, 0])
