import numpy as np
import torch
import os
import foolbox as fb

from verinet.parsers.onnx_parser import ONNXParser

num_images = 100  # number of MNIST images to be loaded
epsilon = 0.015  # For the epsilon, use the values from the Batten paper
dataset = "mnist"
nn_file = "mnist_relu_4_1024.onnx"

# # load single image
# image = np.load(os.path.join("mnist100", "test-{}.npy".format(0)))
# labels = np.loadtxt(os.path.join("mnist100", "all_labels"), dtype=int)[:1]
# image = torch.tensor(np.swapaxes(image, 0, 1))  # Pytorch wants (batch_size, dimensions)!!!
# labels = torch.tensor(labels)

# load images and labels
images = []
for i in range(num_images):
    images.append(np.load(os.path.join("mnist_dataset", "test-{}.npy".format(i))))
images = np.squeeze(np.stack(images, axis=0))  # join images in numpy array, remove last axis
images = torch.tensor(images)
labels = np.loadtxt(os.path.join("mnist_dataset", "all_labels"), dtype=int)
labels = torch.tensor(labels[:num_images])

# load VeriNet model
path = os.path.join(dataset, nn_file)  # path to NN file
parser = ONNXParser(path)
model = parser.to_pytorch()


# convert VeriNet model to a torch model (using the same layers)
# this is necessary because Foolbox cannot process the original VeriNet model correctly
class Network(torch.nn.Module):
    def __init__(self, verinet_model):
        super().__init__()
        self.layers = []
        for layer in verinet_model.layers:
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x
# class Network(torch.nn.Module):
#     def __init__(self, verinet_model):
#         super().__init__()
#         self.layers = []
#         for layer in verinet_model.layers:
#             if isinstance(layer, torch.nn.modules.linear.Identity):
#                 self.layers.append(torch.nn.modules.linear.Identity())
#             elif isinstance(layer, torch.nn.modules.flatten.Flatten):
#                 self.layers.append(torch.nn.modules.flatten.Flatten(layer.start_dim, layer.end_dim))
#             elif isinstance(layer, torch.nn.modules.linear.Linear):
#                 new_layer = torch.nn.modules.linear.Linear(in_features=layer.weight.shape[1],
#                                                            out_features=layer.weight.shape[0])
#                 new_layer.weight = layer.weight
#                 new_layer.bias = layer.bias
#                 self.layers.append(new_layer)
#             elif isinstance(layer, torch.nn.modules.activation.ReLU):
#                 self.layers.append(torch.nn.modules.activation.ReLU(inplace=layer.inplace))
#             else:
#                 raise NotImplementedError("Processing for layer of type " + str(layer) + " not implemented")
#
#     def forward(self, x):
#         for layer in self.layers:
#             x = layer(x)
#
#         return x


pytorch_model = Network(model)

# build Foolbox model and execute attack
fmodel = fb.PyTorchModel(pytorch_model.eval(), bounds=(0, 1))

attack = fb.attacks.LinfDeepFoolAttack()

raw, clipped, is_adv = attack(fmodel, images, labels, epsilons=epsilon)
# raw: raw adversarial examples, may not be true adversarial examples because epsilon is not respected
# clipped: adv. examples clipped to be disturbed by at most epsilon, may not be true adv. examples because of clipping
# is_adv: array yielding whether each clipped adversarial example is a

n_success = np.sum(is_adv.numpy())
print("Number of successful attacks: {}".format(n_success))
print("Upper Bound for % verified: {}%".format(100 - n_success/num_images * 100))
