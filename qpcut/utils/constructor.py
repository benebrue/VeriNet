import gurobipy as gp
from gurobipy import GRB
import numpy as np
import typing
import time


def get_quadratic_coefficients(lpre: typing.List[np.ndarray],
                               upre: typing.List[np.ndarray],
                               layer: int) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Remember that pre-activation bounds for layer i are in lpre[i-1]
    unstable = ~(lpre[layer-1] > 0) & ~(upre[layer-1] < 0)  # not stably active or inactive --> unstable
    qcvalid = (np.abs(lpre[layer-1]) >= upre[layer-1])  # quadratic cuts are valid
    qccand = unstable & qcvalid  # candidates for quadratic cuts
    qccand_ind = np.where(qccand)[0]  # indices for which a valid quadratic cut could be added, [0] because this returns
    # a tuple containing the array of indices at which the condition was met
    lpre_cand = lpre[layer-1][qccand_ind]
    upre_cand = upre[layer-1][qccand_ind]
    print("Layer {} with {} neurons, unstable: {}, with valid quadratic cuts: {}".format(
        layer, lpre[layer-1].shape[0], unstable.sum(), qccand.sum())
    )
    # Calculate coefficients for a1 x^2 + a2 x + a3 for candidate indices (those for which the cuts are valid)
    # derived from g(l)=0, g(u)=u and g'(l)=0
    a1 = upre_cand / (upre_cand - lpre_cand)**2
    a2 = (-2 * lpre_cand * upre_cand) / (upre_cand - lpre_cand)**2
    a3 = (upre_cand**2 * (2 * lpre_cand - upre_cand)) / (upre_cand - lpre_cand)**2 + upre_cand

    return qccand_ind, a1, a2, a3


def construct_model(wts: typing.List[np.ndarray],
                    bias: typing.List[np.ndarray],
                    sizes: np.ndarray,
                    lpost: typing.List[np.ndarray],
                    upost: typing.List[np.ndarray],
                    lpre: typing.List[np.ndarray],
                    upre: typing.List[np.ndarray],
                    n_layers: int,
                    input_image: np.ndarray,
                    add_quadratic_cuts: bool,
                    max_cuts_per_layer: int) -> typing.Tuple[gp.Model, list, list]:
    """
    Construct a MIQP for NN verification using the cuts from the Kochdumper paper
    """
    try:
        # Create new model
        m = gp.Model("qpverifier")

        # Create variables
        x = []  # neurons
        d = [None]  # binary vars, no entry for the first layer

        x.append(m.addMVar(shape=sizes[0], vtype=GRB.CONTINUOUS, name="x_" + str(0)))

        # Input constraints
        m.addConstr(x[0] >= lpost[0], "inp_lower")
        m.addConstr(x[0] <= upost[0], "inp_upper")

        # Input constraints for testing
        # m.addConstr(x[0] >= input_image, "inp_image_lower")
        # m.addConstr(x[0] <= input_image, "inp_image_upper")

        for layer in range(1, n_layers):  # last layer is output layer so e.g. for 1 hidden 1 output n_layers=2
            x.append(m.addMVar(shape=sizes[layer], vtype=GRB.CONTINUOUS, name="x_" + str(layer)))
            # d.append(m.addMVar(shape=sizes[layer], vtype=GRB.BINARY, name="d_" + str(layer)))

            # Use LP relaxation instead
            d.append(m.addMVar(shape=sizes[layer], vtype=GRB.CONTINUOUS, name="d_" + str(layer)))
            m.addConstr(d[layer] >= np.zeros((sizes[layer],)), "lower_binary_" + str(layer))
            m.addConstr(d[layer] <= np.ones((sizes[layer],)), "upper_binary_" + str(layer))

            # ReLU Encoding Constraint 1: x_{i+1} >= W_i x_i + b_i
            m.addConstr(x[layer] - wts[layer-1] @ x[layer-1] >= bias[layer-1], "relax_1_l_" + str(layer))

            # ReLU Encoding Constraint 2: x_{i+1} <= W_i x_i + b_i - l_i * (1 - delta_i)
            # l_i * (1 - delta_i) is a pointwise product so use * instead of @
            m.addConstr(x[layer] - wts[layer-1] @ x[layer-1] + lpre[layer-1] * (
                    np.ones_like(lpre[layer-1]) - d[layer]) <= bias[layer-1], "relax_2_l_" + str(layer))

            # ReLU Encoding Constraint 3: x_{i+1} <= u_i delta_i
            m.addConstr(x[layer] <= upre[layer-1] * d[layer], "relax_3_l_" + str(layer))

            # Nonnegativity Constraint: x_{i+1} >= 0
            # We don't need these because Gurobi automatically assumes that all continuous vars are nonnegative
            # To check this: Update model and print lower bounds of "x" variables
            # m.update()
            # print(x[layer].getAttr(GRB.Attr.LB))

            # Quadratic Constraint: x_{i+1} <= a1 (W_i x_i + b_i)^2 + a2 (W_i x_i + b_i) + a3 which strengthens the LP
            # relaxation
            if add_quadratic_cuts:
                m.setParam(GRB.Param.NonConvex, 2)  # accept nonconvex quadratic constraints
                qccand_ind, a1, a2, a3 = get_quadratic_coefficients(lpre, upre, layer)
                n_to_add = min(len(qccand_ind), max_cuts_per_layer)
                for (count, j) in enumerate(qccand_ind[:n_to_add, ]):
                    m.addConstr(
                        x[layer][j]
                        - a1[count] * x[layer-1] @ np.outer(wts[layer-1][j, :], wts[layer-1][j, :]) @ x[layer-1]
                        - (2 * a1[count] * bias[layer-1][j] + a2[count]) * wts[layer-1][j, :] @ x[layer-1]
                        <= a1[count] * bias[layer-1][j]**2 + a2[count] * bias[layer-1][j] + a3[count],
                        "quadratic_l_{}_n_{}".format(layer, j)
                    )
                # add all quadratic constraints at once (not significantly faster)
                # m.addConstrs(
                #     (x[layer][qccand_ind[count]]
                #      - a1[count] * x[layer - 1] @ np.outer(wts[layer - 1][qccand_ind[count], :],
                #                                            wts[layer - 1][qccand_ind[count], :]) @ x[layer - 1]
                #      - (2 * a1[count] * bias[layer - 1][qccand_ind[count]] + a2[count]) *
                #      wts[layer - 1][qccand_ind[count], :] @ x[layer - 1]
                #      <= a1[count] * bias[layer - 1][qccand_ind[count]] ** 2 +
                #      a2[count] * bias[layer - 1][qccand_ind[count]] + a3[count]
                #      for count in range(n_to_add)),
                #     name="quadratic_l_{}".format(layer)
                # )

    except gp.GurobiError as e:
        print("Error Code: " + str(e.errno) + ": " + str(e))
        raise  # reraise exception, using raise without passing an exception preserves original traceback

    except AttributeError:
        print("Attribute Error encountered")
        raise  # reraise exception, using raise without passing an exception preserves original traceback

    return m, x, d
