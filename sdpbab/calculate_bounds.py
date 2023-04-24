import os
import torch

from verinet.parsers.onnx_parser import ONNXParser
from verinet.sip_torch.rsip import RSIP
from verinet.neural_networks.verinet_nn import VeriNetNN


class BCalculator:

    def __init__(self,
                 model: VeriNetNN,
                 input_shape: list):
        self.model = model
        self.input_shape = input_shape

        """
        Args:
            filepath:
                The path of the onnx file
            input_shape:
                The shape of the inputs that the NN processes
        """
    def calculate_bounds(self,
                         input_bounds: torch.Tensor,
                         layer: int = 0) -> torch.Tensor:

        """
        Calculates the bounds of the nodes in the neural network using RSIP

        Args:
            input_bounds:
                The bounds of the input to the neural network as an Nx2 torch.Tensor where N is the input dimension
                (e.g. 1x28x28 for MNIST)
            layer:
                Specifies the layer for which the concrete pre-activation bounds should be returned (default: 0)

        Returns:
            The pre-activation bounds of the nodes in the i-th layer.
        """

        # self.model.set_device(use_gpu=True)

        # create the input_shape tensor as int64 because a torch.LongTensor is required by the RSIP class
        rsip = RSIP(self.model, torch.tensor(self.input_shape, dtype=torch.int64), use_pbar=False)
        rsip.calc_bounds(input_bounds, from_node=0)
        concrete_bounds = rsip.get_bounds_concrete_pre(layer)
        # for node in rsip.nodes:
        #     print(node)

        return concrete_bounds[0]  # since the function returns a list


if __name__ == "__main__":
    path = os.path.join("mnist", "mnist_relu_3_50.onnx")
    original_shape = [1, 28, 28]  # input shape for the MNIST dataset
    # path = os.path.join("cifar10", "ffnnRELU__Point_6_500.onnx")
    # original_shape = [3, 32, 32]  # input shape for the MNIST dataset
    flattened_shape = [original_shape[0] * original_shape[1] * original_shape[2]]
    parser = ONNXParser(path)
    model = parser.to_pytorch()
    calculator = BCalculator(model, original_shape)

    lower_bounds = torch.zeros(flattened_shape)
    upper_bounds = torch.ones(flattened_shape)
    inp_bounds = torch.stack((lower_bounds, upper_bounds), dim=-1)
    bounds = calculator.calculate_bounds(inp_bounds, layer=3)
    print(bounds)
    print(bounds.shape)

    # df = pd.read_csv("filenames.csv")
    # for idx, row in df.iterrows():
    #     if row["dataset"] == "MNIST":
    #         path = os.path.join("mnist", row["filename"])
    #         original_shape = [1, 28, 28]
    #     else:
    #         path = os.path.join("cifar10", row["filename"])
    #         original_shape = [3, 32, 32]
    #     print("Current Network: " + row["filename"])
    #     flattened_shape = [original_shape[0] * original_shape[1] * original_shape[2]]
    #     calculator = BCalculator(path, original_shape)
    #
    #     lower_bounds = torch.zeros(flattened_shape)
    #     upper_bounds = torch.ones(flattened_shape)
    #     inp_bounds = torch.stack((lower_bounds, upper_bounds), dim=-1)
    #     bounds = calculator.calculate_bounds(inp_bounds, layer=3)
    #     print(bounds[0])
    #     print(bounds[0].shape)
