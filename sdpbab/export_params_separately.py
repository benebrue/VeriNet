"""
This script computes the pre-activation bounds for all layers in a feed-forward NN and stores them together with the
weights and biases. The input bounds are derived from the Batten (2021) paper and the VeriGauge GitHub Repo, the MNIST
images are loaded from the original Raghunathan files while the CIFAR10 images are loaded using torchvision
The resulting data is exported to separate .mat files which are in the format that the LayerSDP script requires
"""

import os
import torch
import torchvision
from scipy.io import savemat
import numpy as np

from verinet.parsers.onnx_parser import ONNXParser
from calculate_bounds import BCalculator

num_images = 100

# IMPORTANT: Change the epsilon according to the specifications
epsilon = 0.1  # For the epsilon, use the values from the Batten paper:
# ETH: 6x100: 0.026, 9x100: 0.026, 6x200: 0.015, 9x200: 0.015
# Raghunathan: nips_lp, nips_pgd, nips_sdp: 0.1
# VeriGauge: cifar10_A_certadv_2255, cifar10_B_certadv_2255: 2/255
# VeriGauge: mnist_G_certadv_1: 0.1, mnist_G_certadv_2: 0.3
# Jianglin's self-trained MNIST network: 0.05
# alpha-beta-CROWN: mnist_cnn_a_adv: 0.3, mnist_conv_small_nat: 0.12
dataset = "mnist"  # currently implemented: "mnist", "cifar10"
network_folder = "small_mnist"  # mnist, small_mnist, self_trained_2_50_0.05, verigauge, alpha_beta_crown
nn_file = "nips_lp.onnx"
path = os.path.join(network_folder, nn_file)  # path to NN file
if dataset == "mnist":
    original_shape = [1, 28, 28]  # input shape for the network
    labels = np.loadtxt(os.path.join("mnist_dataset", "all_labels"))
    labels = labels.astype(int)
    labels = torch.tensor(labels[:num_images])
elif dataset == "cifar10":
    original_shape = [3, 32, 32]
    cifar10_dataset = torchvision.datasets.CIFAR10(root="cifar10_dataset", train=False, download=True,
                                                   transform=torchvision.transforms.Compose([
                                                       torchvision.transforms.ToTensor()
                                                   ]))
    cifar10_loader = iter(torch.utils.data.DataLoader(cifar10_dataset, batch_size=1, shuffle=False))
else:
    raise NotImplementedError("Dataset " + dataset + " is unknown")
flattened_shape = [original_shape[0] * original_shape[1] * original_shape[2]]

parser = ONNXParser(path)
model = parser.to_pytorch()
calculator = BCalculator(model, original_shape)

for image in range(num_images):
    export_path = os.path.join("data", "export", nn_file[:-5] + "_{}".format(epsilon), "image_{}".format(image))
    if not os.path.exists(export_path):  # create export directory
        os.makedirs(export_path)
    if dataset == "mnist":
        input_image = torch.tensor(np.load(os.path.join(dataset + "_dataset", "test-{}.npy".format(image))))
        label = labels[image]
    elif dataset == "cifar10":
        next_item = next(cifar10_loader)
        input_image = next_item[0]
        label = next_item[1]
    else:
        raise NotImplementedError("Dataset " + dataset + " is unknown")

    lower_bounds = torch.flatten(input_image) - epsilon
    upper_bounds = torch.flatten(input_image) + epsilon
    inp_bounds = torch.stack((lower_bounds, upper_bounds), dim=-1)

    counter = 0
    lb_pre = []
    ub_pre = []
    weight_list = []
    bias_list = []
    for idx, node in enumerate(model.nodes):
        if isinstance(node.op, torch.nn.modules.linear.Linear) or isinstance(node.op, torch.nn.Conv2d):
            # idx+1 because this calculates the pre-layer bounds while we want the bounds after the linear layer
            bounds = calculator.calculate_bounds(inp_bounds, layer=idx+1).numpy()
            weights = node.op.weight.detach().numpy()
            biases = node.op.bias.detach().numpy()

            # save data in a dict
            counter = counter + 1
            lb_pre.append(bounds[:, 0])
            ub_pre.append(bounds[:, 1])
            weight_list.append(weights)
            bias_list.append(biases)

    # save everything as separate Matlab files
    print("Saving Files for Image {}".format(image))
    # L
    savemat(os.path.join(export_path, "L.mat"), {"L": float(counter)})
    # Bias
    Bias = np.empty((1, counter), dtype=object)
    for i in range(counter):
        Bias[0, i] = bias_list[i].reshape(-1, 1).astype(np.double)
    savemat(os.path.join(export_path, "Bias.mat"), {"Bias": Bias})
    # Lpost  --> post activation bounds, so need to apply ReLU function
    Lpost = np.empty((1, counter), dtype=object)
    Lpost[0, 0] = np.maximum(0, lower_bounds.numpy()).reshape(-1, 1).astype(np.double)  # post-activation bounds of zero-th layer = input bounds
    for i in range(1, counter):
        Lpost[0, i] = np.maximum(0, lb_pre[i-1]).reshape(-1, 1).astype(np.double)
    savemat(os.path.join(export_path, "Lpost.mat"), {"Lpost": Lpost})
    # Lpre
    Lpre = np.empty((1, counter-1), dtype=object)
    for i in range(counter-1):
        Lpre[0, i] = lb_pre[i].reshape(-1, 1).astype(np.double)  # apply ReLU
    savemat(os.path.join(export_path, "Lpre.mat"), {"Lpre": Lpre})
    # sizes
    sizes = [bias_list[i].shape[0] for i in range(len(bias_list))]
    sizes.insert(0, weight_list[0].shape[1])
    savemat(os.path.join(export_path, "sizes.mat"), {"sizes": np.array(sizes, dtype=np.double)})
    # Upost
    Upost = np.empty((1, counter), dtype=object)
    Upost[0, 0] = np.maximum(0, upper_bounds.numpy()).reshape(-1, 1).astype(np.double)  # post-activation bounds of zero-th layer = input bounds
    for i in range(1, counter):
        Upost[0, i] = np.maximum(0, ub_pre[i-1]).reshape(-1, 1).astype(np.double)
    savemat(os.path.join(export_path, "Upost.mat"), {"Upost": Upost})
    # Upre
    Upre = np.empty((1, counter - 1), dtype=object)
    for i in range(counter - 1):
        Upre[0, i] = ub_pre[i].reshape(-1, 1).astype(np.double)  # apply ReLU
    savemat(os.path.join(export_path, "Upre.mat"), {"Upre": Upre})
    # Weights
    Wts = np.empty((1, counter), dtype=object)
    for i in range(counter):
        Wts[0, i] = weight_list[i].astype(np.double)
    savemat(os.path.join(export_path, "Wts.mat"), {"Wts": Wts})
    # True Label
    savemat(os.path.join(export_path, "Label.mat"), {"Label": label.numpy().item()})
    # Image itself
    savemat(os.path.join(export_path, "Input_image.mat"), {"Input_image": input_image.numpy()})
