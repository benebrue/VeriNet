"""
Uses Keras to load a .pb model and export its weights, biases and layer types as .npy files.
These files can then be used to convert the model to onnx by using the npy_to_onnx.py script.
Important to know: for a layer with m inputs and n outputs, the weight matrix in Keras has shape m*n --> needs to be
transposed before saving since Pytorch expects the reverse formatting.
"""
import os
import numpy as np
import tensorflow.keras as keras

path = "self_trained_2_50_0.05"
net_name = "self_trained_mnist"
model = keras.models.load_model(path)
print(model.summary())

net_layer_types = []
count = 0
for layer in model.layers:
    config = layer.get_config()
    if "activation" in config:  # should be a dense layer in that case
        if config["activation"] == "relu":  # dense layer with ReLU activation
            net_layer_types.append("ff_relu")
            np.save(os.path.join(path, "numpy", net_name + "_w_" + str(count) + ".npy"),
                    np.transpose(layer.get_weights()[0]))
            np.save(os.path.join(path, "numpy", net_name + "_b_" + str(count) + ".npy"), layer.get_weights()[1])
            count += 1
        elif config["activation"] == "softmax":  # last layer
            net_layer_types.append("ff")
            np.save(os.path.join(path, "numpy", net_name + "_w_" + str(count) + ".npy"),
                    np.transpose(layer.get_weights()[0]))
            np.save(os.path.join(path, "numpy", net_name + "_b_" + str(count) + ".npy"), layer.get_weights()[1])
            count += 1
        else:
            raise NotImplementedError("Processing for layer " + config["name"] + " with activation " +
                                      config["activation"] + " not implemented!")

np.save(os.path.join(path, "numpy", net_name + "_layers.npy"), net_layer_types)

print(net_layer_types)
