import torch


class SpikeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, layer_input):
        ctx.save_for_backward(layer_input)
        out = torch.zeros_like(layer_input)
        out[layer_input > 0] = 1.0  # We spike when the (potential-threshold) > 0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        forward_input, = ctx.saved_tensors
        grad_input = grad_output.clone()  # Clone will create a copy of the numerical value
        grad_input[forward_input < 0] = 0  # The derivative of a ReLU function
        return grad_input
