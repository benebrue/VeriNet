import matlab.engine
import numpy as np
import typing


class MatlabParser:
    def __init__(self, path: str):
        """
        Args:
            path: The path at which the Matlab SDP code can be found
        """
        self.eng = matlab.engine.start_matlab()
        self.matlab_path = path
        self.eng.cd(self.matlab_path, nargout=0)

    def solve_layersdp(self,
                       weight_list: typing.List[np.ndarray],
                       bias_list: typing.List[np.ndarray],
                       lower_in_bounds: np.ndarray,
                       upper_in_bounds: np.ndarray,
                       lpre_list: typing.List[np.ndarray],
                       upre_list: typing.List[np.ndarray],
                       true_label: int,
                       attack_label: int) -> typing.Tuple[float, float]:
        """
        Solve LayerSDP using Matlab and return the results to Python.
        Args:
            weight_list: List of weights for the neural network as numpy arrays
            bias_list: List of biases for the neural network as numpy arrays, shape (n_nodes, )
            lower_in_bounds: Lower input bounds for verification as a numpy array, shape (n_input_nodes, )
            upper_in_bounds: Upper input bounds for verification as a numpy array, shape (n_input_nodes, )
            lpre_list: List of lower pre-activation bounds for all layers as numpy arrays, shape (n_nodes, )
            upre_list: List of upper pre-activation bounds for all layers as numpy arrays, shape (n_nodes, )
            true_label: True label for the current input as an int
            attack_label: Label against which robustness is verified as an int
        Returns:
            objective: Optimal value found by the SDP verifier, the network is robust if this is > 0
            runtime: Time required for solving the SDP
        """

        # Construct inputs for load_solve_LayerSDP() as lists since according to the Matlab documentation at
        # https://uk.mathworks.com/help/matlab/matlab_external/pass-data-to-matlab-from-python.html lists as Python
        # containers are automatically mapped to Matlab Cell arrays
        print("Constructing inputs...")
        n_layers = len(weight_list)
        # Bias
        Bias = []
        for i in range(n_layers):
            Bias.append(bias_list[i].reshape(-1, 1).astype(np.double))
        # Lpost  --> post activation bounds, so need to apply ReLU function
        # post-activation bounds of zero-th layer = input bounds
        Lpost = [np.maximum(0, lower_in_bounds).reshape(-1, 1).astype(np.double)]
        for i in range(1, n_layers):
            Lpost.append(np.maximum(0, lpre_list[i - 1]).reshape(-1, 1).astype(np.double))
        # Lpre
        Lpre = []
        for i in range(n_layers - 1):
            Lpre.append(lpre_list[i].reshape(-1, 1).astype(np.double))  # apply ReLU
        # sizes
        sizes = [float(bias_list[i].shape[0]) for i in range(len(bias_list))]
        sizes.insert(0, float(weight_list[0].shape[1]))
        sizes = np.array(sizes, dtype=np.double)
        # Upost
        # post-activation bounds of zero-th layer = input bounds
        Upost = [np.maximum(0, upper_in_bounds).reshape(-1, 1).astype(np.double)]
        for i in range(1, n_layers):
            Upost.append(np.maximum(0, upre_list[i - 1]).reshape(-1, 1).astype(np.double))
        # Upre
        Upre = []
        for i in range(n_layers - 1):
            Upre.append(upre_list[i].reshape(-1, 1).astype(np.double))  # apply ReLU
        # Weights
        Wts = []
        for i in range(n_layers):
            Wts.append(weight_list[i].astype(np.double))

        final_weight = Wts[-1][true_label, :] - Wts[-1][attack_label, :]
        final_bias = Bias[-1][true_label, 0] - Bias[-1][attack_label, 0]

        # call actual verifier using the MATLAB engine, use nargout to explicitly get both return values
        # Do we need to do this asynchronously?
        # https://uk.mathworks.com/help/matlab/matlab_external/call-matlab-functions-asynchronously-from-python.html
        objective, runtime = self.eng.solve_LayerSDP(Wts, Bias, sizes, Upost, Lpost, Upre, Lpre, final_weight,
                                                     final_bias, nargout=2)

        return objective, runtime
