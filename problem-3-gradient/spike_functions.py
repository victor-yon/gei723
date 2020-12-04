import torch
import numpy as np


class SpikeFunctionRelu(torch.autograd.Function):
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


class SpikeFunctionFastSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, layer_input):
        ctx.save_for_backward(layer_input)
        out = torch.zeros_like(layer_input)
        out[layer_input > 0] = 1.0  # We spike when the (potential-threshold) > 0 #raise NotImplemented
        return out

    @staticmethod
    def backward(ctx, grad_output):
        forward_input, = ctx.saved_tensors
        grad_input = grad_output.clone()  # Clone will create a copy of the numerical value
        grad_input = grad_input/(SpikeFunctionRelu.scale*torch.abs(input)+1.0)**2
        return grad_input


class SpikeFunctionPiecewise(torch.autograd.Function):
    @staticmethod
    def forward(ctx, layer_input):
        ctx.save_for_backward(layer_input)
        out = torch.zeros_like(layer_input)
        out[layer_input > 0] = 1.0  # We spike when the (potential-threshold) > 0 #raise NotImplemented
        return out

    @staticmethod
    def backward(ctx, grad_output):
        forward_input, = ctx.saved_tensors
        grad_input = grad_output.clone()  # Clone will create a copy of the numerical value
        grad_input[np.where(forward_input < -0.5)] = 0  # segments between [0, 0.5], [0.5, 1], [1+] with -1 because of voltage threshold
        grad_input[np.where((forward_input > -0.5) & (forward_input < 0))] = 2
        grad_input[np.where(forward_input >= 0)] = -2
        return grad_input
