import numpy as np


def propagate_input(w, b, final_weight, final_bias, input_image):
    output = input_image
    n_layers = len(w)
    for i in range(n_layers-1):
        output = w[i].dot(output) + b[i]
        if i < n_layers:
            output = np.maximum(np.zeros_like(output), output)
    output = final_weight.dot(output) + final_bias

    return output.item()
