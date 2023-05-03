import os
import sys
import pickle
import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB

# manually add VeriNet root directory to the $PYTHONPATH
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print(SCRIPT_DIR)
sys.path.append(SCRIPT_DIR)
# print(sys.path)

from qpcut.utils.compressor import compress_nn
from qpcut.utils.constructor import construct_model
from qpcut.utils.propagate_input import propagate_input

dataset = "nips_sdp"
epsilon = 0.1
n_samples = 100
n_classes = 10
add_quadratic_cuts = True

base_path = os.path.join("data", "export", dataset + "_" + str(epsilon))
total_time = 0
n_robust = 0

for i in range(n_samples):
    # Load data from .pkl files and use np.squeeze() to transform all arrays of shape (n, 1) to shape (n, ) since (n, 1)
    # arrays cause problems in Gurobi
    input_path = os.path.join(base_path, "image_" + str(i))
    with open(os.path.join(input_path, "bias.pkl"), "rb") as f:
        bias = pickle.load(f)
        bias = [np.squeeze(b) for b in bias]
    with open(os.path.join(input_path, "input_image.pkl"), "rb") as f:
        input_image = np.squeeze(pickle.load(f))
    with open(os.path.join(input_path, "label.pkl"), "rb") as f:
        label = pickle.load(f)
    with open(os.path.join(input_path, "lpost.pkl"), "rb") as f:
        lpost = pickle.load(f)
        lpost = [np.squeeze(lp) for lp in lpost]
    with open(os.path.join(input_path, "lpre.pkl"), "rb") as f:
        lpre = pickle.load(f)
        lpre = [np.squeeze(lp) for lp in lpre]
    with open(os.path.join(input_path, "n_layers.pkl"), "rb") as f:
        n_layers = pickle.load(f)
    with open(os.path.join(input_path, "sizes.pkl"), "rb") as f:
        sizes = pickle.load(f)
    with open(os.path.join(input_path, "upost.pkl"), "rb") as f:
        upost = pickle.load(f)
        upost = [np.squeeze(up) for up in upost]
    with open(os.path.join(input_path, "upre.pkl"), "rb") as f:
        upre = pickle.load(f)
        upre = [np.squeeze(up) for up in upre]
    with open(os.path.join(input_path, "wts.pkl"), "rb") as f:
        wts = pickle.load(f)

    attack_labels = [idx for idx in range(n_classes) if idx != label]

    # print("Output before NNcompression: {}".format(propagate_input(
    #     wts, bias, wts[-1][label, :] - wts[-1][attack_labels[0], :], bias[-1][label] - bias[-1][attack_labels[0]],
    #     input_image
    # )))
    wts, bias, sizes, lpost, upost, lpre, upre, n_layers, _, __ = compress_nn(wts, bias, sizes, lpost, upost, lpre,
                                                                              upre, n_layers)
    # print("Output after NNcompression: {}".format(propagate_input(
    #     wts, bias, wts[-1][label, :] - wts[-1][attack_labels[0], :], bias[-1][label] - bias[-1][attack_labels[0]],
    #     input_image
    # )))

    # Construct model once for each i, then only change the objective for each j and reoptimise
    construction_start = time.time()
    m, x, d = construct_model(wts, bias, sizes, lpost, upost, lpre, upre, n_layers, input_image, add_quadratic_cuts)
    image_time = time.time() - construction_start
    print("Model constructed in {:.2f} seconds".format(image_time))
    is_robust = True

    print("####################### Image {} #######################".format(i))

    for j in attack_labels:
        # Remember that wts, bias, lpost, upost, lpre, upre are lists
        try:
            # Set objective
            final_weight = wts[-1][label, :] - wts[-1][j, :]
            final_bias = bias[-1][label] - bias[-1][j]
            m.setObjective(final_weight @ x[-1] + final_bias, GRB.MINIMIZE)  # objective value > 0 --> network robust

            # Optimise model
            # m.write("before.lp")  # output model to file
            # m.setParam("OutputFlag", 0)  # disable verbose solver logging
            optimisation_start = time.time()
            m.optimize()
            t0 = time.time() - optimisation_start
            image_time += t0
            # For debugging: Compute Irreducible Inconsistent Subsystem and write it to a .ilp file
            # m.computeIIS()
            # m.write("model.ilp")
            # Print results
            if m.ObjVal > 0:
                print("j={}: objective={:.2f} --> {}".format(j, m.ObjVal, "Robust"))
            else:
                print("j={}: objective={:.2f} --> {}".format(j, m.ObjVal, "Not robust"))
                is_robust = False
                break  # not robust for j-th label --> network not robust for image at hand

            # Compare with manual computation (for testing purposes)
            # result = propagate_input(wts, bias, final_weight, final_bias, input_image)
            # print("Manual computation yielded: " + str(result))
            # print("Difference small for i={}, j={}: {}".format(i, j, str(abs(m.ObjVal - result) < 0.01)))

        except gp.GurobiError as e:
            print("Error Code: " + str(e.errno) + ": " + str(e))
            raise

        except AttributeError:
            print("Attribute Error encountered")
            raise

    print("Network robust for image {}: {}".format(i, is_robust))
    print("Time required for that image: {:.2f}".format(image_time))
    total_time += image_time
    if is_robust:
        n_robust += 1

print("####################### Verification complete #######################")
print("Dataset {} with epsilon={} is robust for {} samples".format(dataset, epsilon, n_robust))
print("Average time required for verifying a sample: {:.2f} seconds".format(total_time/n_samples))
