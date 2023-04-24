import torch
import os
import pickle
import numpy as np
from verinet.parsers.onnx_parser import ONNXParser

folder = "small_mnist"  # small_mnist, self_trained_2_50_0.05, verigauge
net_name = "nips_sdp"  # nips_lp, nips_pgd, nips_sdp, self_trained_mnist, cifar10_A_certadv_2255

# Load model using VeriNet
path = os.path.join(folder, net_name + ".onnx")
parser = ONNXParser(path)
model = parser.to_pytorch()


# Convert to PyTorch model
params = []
for node in model.nodes:
    if isinstance(node.op, torch.nn.modules.linear.Linear):
        weight = node.op.weight.detach().numpy().T  # transpose since models have (out_dim, in_dim) weights but
        # jax_verify expects (in_dim, out_dim) weights
        bias = node.op.bias.detach().numpy()
        params.append((weight, bias))

for param in params:
    print(param[0].shape)
    print(param[1].shape)

with open(os.path.join(folder, net_name + ".pkl"), "wb") as f:
    pickle.dump(params, f)

