import torchvision.datasets
import torch
import os
import pickle

# This script loads the MNIST dataset using the corresponding torch function and export the first couple images as
# numpy arrays using pickle

mnist_trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                            transform=torchvision.transforms.ToTensor())
labels = []
for i in range(10):
    print(mnist_trainset[i][1])
    with open(os.path.join("data", "save", "mnist_{}.pickle".format(i)), "wb") as output_file:
        pickle.dump(mnist_trainset[i][0].numpy(), output_file)
        labels.append(mnist_trainset[i][1])

with open(os.path.join("data", "save", "labels.pickle"), "wb") as label_file:
    pickle.dump(labels, label_file)

loaded = torch.load(os.path.join("data", "save", "mnist_0"))
print("Loaded")
print(loaded)
