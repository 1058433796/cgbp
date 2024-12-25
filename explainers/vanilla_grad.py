import torch
from utils.image_process import min_max_normalize


class VanillaGrad:
    def __init__(self, model, target_layer=None, use_raw_output_for_gradient=True):
        self.model = model
        self.use_raw_output_for_gradient = use_raw_output_for_gradient
        self.target_layer = target_layer

    def generate_hm(self, inp_tensor, target_class):
        grad = self.get_grad(inp_tensor, target_class)
        hm = grad.sum((0, 1)).abs()
        hm = min_max_normalize(hm)
        return hm.detach().cpu().numpy()

    def get_grad(self, inp_tensor, target_class):
        inp_tensor = inp_tensor.detach()
        inp_tensor.requires_grad = True

        values = dict()
        handle = None

        if self.target_layer is not None:

            def backward_hook(module, input, output):
                values['gradients'] = output[0]

            handle = self.target_layer.register_backward_hook(backward_hook)

        output = self.model(inp_tensor)
        if self.use_raw_output_for_gradient:
            output[:, target_class].backward()
        else:
            prob = torch.softmax(output, 1)
            prob[:, target_class].backward()

        if self.target_layer is not None:
            grad = values['gradients']
            handle.remove()
        else:
            grad = inp_tensor.grad

        return grad
