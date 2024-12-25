import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from utils.image_process import min_max_normalize
from explainers import VanillaGrad
from torchvision.models import VGG, ResNet
from models.myresnet import MyResnet
class RectGrad:
    def __init__(self, model, target_layer=None, q_percentage=0.98, use_raw_output_for_gradient=True):
        self.model = model
        self.q_percentage = q_percentage
        self.grad_calculator = VanillaGrad(
            model, target_layer=target_layer,
            use_raw_output_for_gradient=use_raw_output_for_gradient
        )
        self.dict = {
            'relu_activations': [],
        }

    def generate_hm(self, inp_tensor, target_class):
        grad = self.get_grad(inp_tensor, target_class)
        hm = grad.sum((0, 1)).abs()
        # hm = grad.abs().sum((0, experiment1))
        hm = min_max_normalize(hm)
        return hm.detach().cpu().numpy()

    def get_hooks(self):
        def relu_backward_hook(module, inp, out):
            activation = self.dict['relu_activations'][-1]
            del self.dict['relu_activations'][-1]

            activation_mul_grad = activation * out[0]
            tau = torch.quantile(activation_mul_grad, self.q_percentage)

            modified_grad_out = torch.where(activation_mul_grad > tau, 1., 0.) * out[0]

            return modified_grad_out,

        def relu_forward_hook(module, inp, out):
            self.dict['relu_activations'].append(out)

        return relu_forward_hook, relu_backward_hook

    def install_hooks(self):
        relu_forward_hook, relu_backward_hook = self.get_hooks()

        handles = []
        if isinstance(self.model, VGG):
            for module in self.model.features.modules():
                if isinstance(module, torch.nn.ReLU):
                    h1 = module.register_backward_hook(relu_backward_hook)
                    h2 = module.register_forward_hook(relu_forward_hook)
                    handles.extend([h1, h2])
        elif isinstance(self.model, ResNet):
            for module in self.model.modules():
                if isinstance(module, torch.nn.ReLU):
                    h1 = module.register_backward_hook(relu_backward_hook)
                    h2 = module.register_forward_hook(relu_forward_hook)
                    handles.extend([h1, h2])
        elif isinstance(self.model, MyResnet):
            for module in self.model.modules():
                if isinstance(module, torch.nn.ReLU) and module not in self.model.model.fc:
                    h1 = module.register_backward_hook(relu_backward_hook)
                    h2 = module.register_forward_hook(relu_forward_hook)
                    handles.extend([h1, h2])
        else:
            raise ValueError("Unsupported model")
        return handles

    def uninstall_hooks(self, handles):
        for handle in handles:
            handle.remove()

    def get_grad(self, input, target_class):
        handles = self.install_hooks()
        grad = self.grad_calculator.get_grad(input, target_class)
        self.uninstall_hooks(handles)
        return grad
