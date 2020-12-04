import torch

ALPHA = 0.001


class SpikeFunctionRelu(torch.autograd.Function):
    alpha = None

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
        grad_input[forward_input < 0] = grad_input[forward_input < 0] * ALPHA  # The derivative of a ReLU function
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
        grad_input = ALPHA * grad_input / (torch.abs(forward_input) + 1.0) ** 2
        return grad_input


class SpikeFunctionSigmoid(torch.autograd.Function):
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
        grad = grad_input * ALPHA * torch.exp(-ALPHA * forward_input) / (
                1 + torch.exp(-ALPHA * forward_input)) ** 2
        return grad


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
        # segment de droite débutant en alpha et allant jusqu'à 1.
        # EQUATION : (1/(1-alpha)))*forward_input -alpha/(1-alpha)
        grad_input[forward_input >= ALPHA] = (1 / (1 - ALPHA)) * forward_input[
            forward_input >= ALPHA] - ALPHA / (1 - ALPHA)
        return grad_input


class SpikeFunctionPiecewiseSymetrique(torch.autograd.Function):
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
        grad_input[forward_input <= -ALPHA] = 0
        grad_input[forward_input > ALPHA] = 0  # le segment vaut 0 pour x<-0.5 et x > 0.5
        return grad_input
