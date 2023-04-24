import os
import torch

from verinet.parsers.onnx_parser import ONNXParser


class NaiveBCalculator:

    def __init__(self,
                 filepath: str,
                 input_shape: list):
        self.filepath = filepath
        self.input_shape = input_shape

        """
        Args:
            filepath:
                The path of the onnx file
            input_shape:
                The shape of the inputs that the NN processes
        """

    def calculate_naive_bounds(self,
                               input_bounds: torch.Tensor,
                               layer: int = 0) -> torch.Tensor:

        """
        Calculates the bounds of the nodes in the neural network's first layer using RSIP

        Args:
            input_bounds:
                The bounds of the input to the neural network as an Nx2 torch.Tensor where N is the input dimension
                (e.g. 1x28x28 for MNIST)
            layer:
                Specifies the layer for which the concrete pre-activation bounds should be returned (default: 0)

        Returns:
            The pre-activation bounds of the nodes in the i-th layer.
        """
        parser = ONNXParser(self.filepath)
        model = parser.to_pytorch()
        lower = input_bounds[:, :, :, 0]
        upper = input_bounds[:, :, :, 1]

        with torch.no_grad():  # disable gradient computation
            for i in range(layer + 1):
                node = model.nodes[i]
                if isinstance(node.op, torch.nn.modules.linear.Identity):
                    # Do Nothing (Identity Layer)
                    pass
                elif isinstance(node.op, torch.nn.modules.flatten.Flatten):
                    lower = torch.flatten(lower)
                    upper = torch.flatten(upper)
                elif isinstance(node.op, torch.nn.modules.linear.Linear):
                    weights = node.op.weight
                    biases = node.op.bias
                    new_lower = 0.5 * (  # formulas taken from the code by Raghunathan et al.
                            torch.matmul(weights, lower + upper)
                            + torch.matmul(torch.abs(weights), lower - upper)
                    ) + biases
                    new_upper = 0.5 * (
                            torch.matmul(weights, lower + upper)
                            + torch.matmul(torch.abs(weights), upper - lower)
                    ) + biases
                    lower = new_lower  # calculate new values before actually updating the variables!
                    upper = new_upper
                elif isinstance(node.op, torch.nn.modules.activation.ReLU):
                    lower = torch.nn.functional.relu(lower)
                    upper = torch.nn.functional.relu(upper)
                else:
                    raise NotImplementedError("Processing for layer " + str(i) + " of type " + str(node.op) +
                                              " not implemented")

        return torch.stack((lower, upper), dim=-1)


if __name__ == "__main__":
    path = os.path.join("mnist", "mnist_relu_3_50.onnx")
    original_shape = [1, 28, 28]
    ncalculator = NaiveBCalculator(path, original_shape)

    lower_bounds = torch.zeros(original_shape)
    upper_bounds = torch.ones(original_shape)
    inp_bounds = torch.stack((lower_bounds, upper_bounds), dim=-1)
    bounds = ncalculator.calculate_naive_bounds(inp_bounds, layer=2)
    print(bounds)
