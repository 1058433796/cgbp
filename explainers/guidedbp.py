import torch
from explainers import VanillaGrad
from utils.image_process import min_max_normalize
from torchvision.models import VGG, ResNet

class GuidedBackprop:
    def __init__(self, model, target_layer=None):
        self.model = model
        self.grad_calculator = VanillaGrad(model, target_layer)

    def generate_hm(self, inp_tensor, target_class):
        guided_gradients = self.get_grad(inp_tensor, target_class)
        hm = guided_gradients.sum((0, 1)).abs()
        hm = min_max_normalize(hm)
        return hm.detach().cpu().numpy()

    def get_grad(self, inp_tensor, target_class):
        activations = []

        def forward_hook(module, input, output):
            activations.append(output)

        def backward_hook(module, input, output):
            activation = activations[-1]
            del activations[-1]

            activation[activation > 0] = 1
            modified_grad_out = activation * torch.clamp(output[0], min=0)
            return (modified_grad_out, )

        handles = []
        for module in self.model.modules():
            if isinstance(module, torch.nn.ReLU):
                forward_handle = module.register_forward_hook(forward_hook)
                backward_handle = module.register_backward_hook(backward_hook)

                handles.extend([forward_handle, backward_handle])
        grad = self.grad_calculator.get_grad(inp_tensor, target_class)

        for handle in handles:
            handle.remove()

        return grad

