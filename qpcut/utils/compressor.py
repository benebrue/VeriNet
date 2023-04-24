import typing
import numpy as np


def compress_nn(
        wts: typing.List[np.ndarray],
        bias: typing.List[np.ndarray],
        sizes: np.ndarray,
        lpost: typing.List[np.ndarray],
        upost: typing.List[np.ndarray],
        lpre: typing.List[np.ndarray],
        upre: typing.List[np.ndarray],
        n_layers: int,
        final_weight: np.ndarray = None,
        final_bias: np.ndarray = None,
        include_final_weight_bias: bool = False
) -> typing.Tuple[
    typing.List[np.ndarray],
    typing.List[np.ndarray],
    np.ndarray,
    typing.List[np.ndarray],
    typing.List[np.ndarray],
    typing.List[np.ndarray],
    typing.List[np.ndarray],
    int,
    np.ndarray,
    np.ndarray
]:
    # print("Starting NNcompression")
    n_hidden = n_layers - 1  # last layer is not a true hidden layer
    for hidden_layer in range(n_hidden):
        stable_ind = []  # indices of stable neurons at the current layer
        n_stable = 0  # number of stable neurons found
        unstable = False  # whether any unstable neurons have been found in the current layer
        to_remove = np.zeros(sizes[hidden_layer+1])

        for i in range(sizes[hidden_layer+1]):
            # Case 1: Neuron is strictly inactive or neurons with Wi(j)=0
            # Neuron i has constant zero output(either because strictly inactive or because L2 norm of weights = 0)
            if upre[hidden_layer][i] <= 0 or np.linalg.norm(wts[hidden_layer][i, :], 2) == 0:
                # Layer ind_layer still has other neurons (more to iterate through or already found stable neurons saved
                # in S or found an unstable neuron)
                if i < sizes[hidden_layer+1] or len(stable_ind) > 0 or unstable:
                    # if norm of weights is zero but bias is nonzero --> bias has to be taken care of
                    if np.linalg.norm(wts[hidden_layer][i, :], 2) == 0 and bias[hidden_layer][i] > 0:
                        # iterate through neurons in following hidden layer and adjust activations there
                        for j in range(sizes[hidden_layer+2]):
                            # % in weight matrix: rows = neurons in following hidden layer, columns = neurons in current
                            # layer
                            bias[hidden_layer+1][j] = bias[hidden_layer+1][j] + wts[hidden_layer+1][j, i]\
                                                    * bias[hidden_layer][i]
                    # otherwise (bias is zero): no need to adjust anything, just delete neuron
                to_remove[i] = 1  # remove neuron i from this hidden_layer

            # Case 2: strictly active
            elif lpre[hidden_layer][i] > 0:  # neuron is strictly active
                wts_stable = wts[hidden_layer][stable_ind, :]  # get weights of all stable neurons identified so far
                # concatenate wts_stable with vector of current neuron i
                wts_stable_i = np.vstack((wts_stable, wts[hidden_layer][i, :]))
                # avoid computing matrix_rank of empty array (throws ValueError)
                stable_rank = np.linalg.matrix_rank(wts_stable) if wts_stable.shape[0] > 0 else 0
                # wts[hidden_layer][i, :] is linearly independent of the stable neurons collected in stable_ind
                if stable_rank != np.linalg.matrix_rank(wts_stable_i):
                    n_stable += 1
                    stable_ind.append(i)
                else:  # wts[hidden_layer][i, :] is linearly dependent on wts_stable
                    alpha = wts[hidden_layer][i, :] @ np.linalg.pinv(wts_stable)  # pinv: Moore-Penrose pseudoinverse
                    for j in range(sizes[hidden_layer+1]):  # adjust activations at layer ind_layer+1
                        # % only update the neurons in stable_ind, the other ones will be dropped anyway
                        weight_effect = np.sum(alpha @ wts[hidden_layer+1][j, i])
                        wts[hidden_layer+1][j, stable_ind] = wts[hidden_layer+1][j, stable_ind] + weight_effect
                        bias_effect = wts[hidden_layer+1][j, i] @ (bias[hidden_layer][i] -
                                                                   alpha @ bias[hidden_layer][stable_ind])
                        bias[hidden_layer+1][j] = bias[hidden_layer+1][j] + bias_effect

                    if include_final_weight_bias and hidden_layer == n_hidden:
                        # update final_weight before deleting the current entry! This is important since the weight of
                        # the merged neuron might be unequal 0, so it has to be taken into account before being deleted
                        final_weight_effect = alpha @ final_weight[1, i]
                        final_weight[1, stable_ind] = final_weight[1, stable_ind] + final_weight_effect
                        final_bias_effect = final_weight[1, i] @ (bias[hidden_layer][i] -
                                                                  alpha @ bias[hidden_layer][stable_ind])
                        final_bias[1, 1] = final_bias[1, 1] + final_bias_effect

                    to_remove[i] = 1  # remove neuron i from this hidden_layer
            else:
                unstable = True  # at least one neuron is unstable

        if np.sum(to_remove) > 0:  # if there exists at least one node that should be removed
            sizes[hidden_layer+1] = sizes[hidden_layer+1] - np.sum(to_remove)
            to_remain = np.where(to_remove == 0)[0]  # np.where returns a tuple so take first entry
            wts[hidden_layer] = wts[hidden_layer][to_remain, :]
            wts[hidden_layer+1] = wts[hidden_layer+1][:, to_remain]
            bias[hidden_layer] = bias[hidden_layer][to_remain]
            lpre[hidden_layer] = lpre[hidden_layer][to_remain]
            upre[hidden_layer] = upre[hidden_layer][to_remain]
            # "hidden_layer+1" because the first entry in lpost/upost are the input bounds
            lpost[hidden_layer+1] = lpost[hidden_layer+1][to_remain]
            upost[hidden_layer+1] = upost[hidden_layer+1][to_remain]
            if include_final_weight_bias and hidden_layer == n_hidden:
                final_weight = final_weight[to_remain]

        # Case 3: All the neurons left at this hidden layer are stable
        if not unstable and hidden_layer < n_hidden:  # only stable neurons left and not the last layer
            n_layers -= 1
            # if there is at least one neuron in the layer: directly connect hidden_layer-1 to hidden_layer+1
            if len(stable_ind) > 0:
                w_new = np.zeros((sizes[hidden_layer], sizes[hidden_layer+2]))  # new weight matrix
                b_new = np.zeros((sizes[hidden_layer+2], 1))  # new bias vector
                for i in range(sizes[hidden_layer+2]):
                    b_new[i] = bias[hidden_layer+1][i] + wts[hidden_layer+1][i, stable_ind] @ bias[hidden_layer][
                        stable_ind]
                    for j in range(sizes[hidden_layer]):
                        w_new[i, j] = wts[hidden_layer+1][i, stable_ind] @ wts[hidden_layer][stable_ind, j]
                # remove hidden layer, replace weight and bias in hidden_layer+1 with w_new and b_new
                for ind_temp in range(hidden_layer, n_hidden):
                    sizes[ind_temp+1] = sizes[ind_temp+2]
                    if ind_temp == hidden_layer:
                        wts[ind_temp] = w_new
                        bias[hidden_layer] = b_new
                    else:
                        wts[ind_temp] = wts[ind_temp+1]
                        bias[ind_temp] = bias[ind_temp+1]
                    lpre[ind_temp] = lpre[ind_temp+1]
                    upre[ind_temp] = upre[ind_temp+1]
                    lpost[ind_temp+1] = lpost[ind_temp+2]
                    upost[ind_temp+1] = upost[ind_temp+2]
            else:
                sizes[hidden_layer+1] = 0  # no nodes remain at this hidden layer, NN output constant "final_bias_new"
                nn_const = bias[hidden_layer+1]  # propagate bias after layer with no more neurons through the network
                for ind_temp in range(hidden_layer+2, n_hidden+1):
                    nn_const = wts[ind_temp] @ nn_const + bias[ind_temp]
                if include_final_weight_bias:
                    final_bias_new = final_weight @ nn_const
                else:
                    raise NotImplementedError("Handling empty layers for include_final_weight_bias not implemented")
                break

    if any(sizes == 0):  # any fully strictly inactive hidden layer (final_bias_new is computed above)
        final_bias = final_bias_new

    # if include_final_weight_bias is false then final_weight/final_bias are just None
    return wts, bias, sizes, lpost, upost, lpre, upre, n_layers, final_weight, final_bias
