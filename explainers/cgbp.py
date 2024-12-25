import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from models.myresnet import MyResnet
from utils.image_process import min_max_normalize
from explainers import VanillaGrad
from torchvision.models import VGG, ResNet

class CGBP:
    def __init__(self, model, target_layer=None, positive_propagation=True, use_raw_output_for_gradient=False, q_percentage=0.98):
        self.model = model
        self.q_percentage = q_percentage
        self.grad_calculator = VanillaGrad(
            model, target_layer=target_layer,
            use_raw_output_for_gradient=use_raw_output_for_gradient
        )
        self.dict = {
            'relu_activations': [],
            'avgpool_inp_activations': [],
            # 'maxpool_activations': [],
        }
        self.positive_propagation = positive_propagation

    def generate_hm(self, inp_tensor, target_class):
        grad = self.get_grad(inp_tensor, target_class)
        if isinstance(grad, tuple): grad = grad[0]
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

        def avgpool_backward_hook(module, inp, out):
            # print('avgpool 反向传播')
            inp_activation = self.dict['avgpool_inp_activations'][-1]
            del self.dict['avgpool_inp_activations'][-1]

            inp_grad = inp[0]
            activation_mul_grad = inp_grad * inp_activation
            P = torch.clamp(activation_mul_grad, min=0).sum((0, 1))
            N = -torch.clamp(activation_mul_grad, max=0).sum((0, 1))

            P_bar = torch.where(P > N, 1., 0.) * P
            N_bar = torch.where(P < N, 1., 0.) * N

            # plt.subplot(1, 4, 1)
            # plt.imshow(P.detach().cpu())
            # plt.axis('off')
            #
            # plt.subplot(1, 4, 2)
            # plt.imshow(P_bar.detach().cpu())
            # plt.axis('off')
            #
            # plt.subplot(1, 4, 3)
            # plt.imshow(N.detach().cpu())
            # plt.axis('off')
            #
            # plt.subplot(1, 4, 4)
            # plt.imshow(N_bar.detach().cpu())
            # plt.axis('off')
            #
            # plt.show()

            P_mask = torch.where(P_bar > P_bar.mean(), 1., 0.)
            N_mask = torch.where(N_bar > N_bar.mean(), 1., 0.)


            if self.positive_propagation:
                tau = torch.quantile(activation_mul_grad, self.q_percentage)
                mask = torch.where(activation_mul_grad > tau, 1., 0.)
                modified_grad_in = inp_grad * P_mask * mask
            else:
                tau = torch.quantile(activation_mul_grad, 1 - self.q_percentage)
                mask = torch.where(activation_mul_grad < tau, 1., 0.)
                modified_grad_in = -inp_grad * N_mask * mask

            # activation_mul_grad = inp_grad * inp_activation
            # cam = activation_mul_grad.sum((0, experiment1))
            # cam[cam < 0] = 0
            # mask = torch.where(cam > 0, experiment1., 0.)

            # plt.subplot(1, 4, 1)
            # plt.imshow(P.detach().cpu())
            # plt.title('P')
            # plt.axis('off')
            # plt.subplot(1, 4, 2)
            # plt.imshow(N.detach().cpu())
            # plt.title('N')
            # plt.axis('off')
            # plt.subplot(1, 4, 3)
            # plt.imshow(P_bar.detach().cpu())
            # plt.title('P_bar')
            # plt.axis('off')
            # plt.subplot(1, 4, 4)
            # plt.imshow(mask.detach().cpu())
            # plt.title('mask')
            # plt.axis('off')
            # plt.show()

            # modified_grad_in = inp_grad * mask
            return modified_grad_in,

        def relu_forward_hook(module, inp, out):
            self.dict['relu_activations'].append(out)

        def avgpool_forward_hook(module, inp, out):
            self.dict['avgpool_inp_activations'].append(inp[0])

        return (relu_forward_hook, relu_backward_hook,
                # maxpool_forward_hook, maxpool_backward_hook,
                avgpool_forward_hook, avgpool_backward_hook,
                )

    def install_hooks(self):
        relu_forward_hook, relu_backward_hook, avgpool_forward_hook, avgpool_backward_hook = self.get_hooks()

        handles = []
        if isinstance(self.model, VGG):
            for module in self.model.features.modules():
                if isinstance(module, torch.nn.ReLU):
                    h1 = module.register_backward_hook(relu_backward_hook)
                    h2 = module.register_forward_hook(relu_forward_hook)
                    handles.extend([h1, h2])

            h1 = self.model.avgpool.register_backward_hook(avgpool_backward_hook)
            h2 = self.model.avgpool.register_forward_hook(avgpool_forward_hook)
            handles.extend([h1, h2])
        elif isinstance(self.model, ResNet):
            for module in self.model.modules():
                if isinstance(module, torch.nn.ReLU):
                    h1 = module.register_backward_hook(relu_backward_hook)
                    h2 = module.register_forward_hook(relu_forward_hook)
                    handles.extend([h1, h2])

            h1 = self.model.avgpool.register_backward_hook(avgpool_backward_hook)
            h2 = self.model.avgpool.register_forward_hook(avgpool_forward_hook)
            handles.extend([h1, h2])
        elif isinstance(self.model, MyResnet):
            for module in self.model.modules():
                if isinstance(module, torch.nn.ReLU) and module not in self.model.model.fc:
                    h1 = module.register_backward_hook(relu_backward_hook)
                    h2 = module.register_forward_hook(relu_forward_hook)
                    handles.extend([h1, h2])

            h1 = self.model.model.avgpool.register_backward_hook(avgpool_backward_hook)
            h2 = self.model.model.avgpool.register_forward_hook(avgpool_forward_hook)
            handles.extend([h1, h2])
        else:
            raise ValueError("Unsupported model")

        # h1 = self.model.features[-experiment1].register_backward_hook(maxpool_backward_hook)
        # h2 = self.model.features[-experiment1].register_forward_hook(maxpool_forward_hook)
        # handles.extend([h1, h2])

        return handles

    def uninstall_hooks(self, handles):
        for handle in handles:
            handle.remove()

    def get_grad(self, input, target_class):
        handles = self.install_hooks()
        grad = self.grad_calculator.get_grad(input, target_class)
        self.uninstall_hooks(handles)
        return grad

# 依据CAM生成positive的结果以及negative的结果，二者归一化后相减
class CCGBP:
    def __init__(self, model, target_layer=None, use_raw_output_for_gradient=False, q_percentage=0.98):
        self.pos_cgbp = CGBP(model, target_layer, True, use_raw_output_for_gradient, q_percentage)
        self.neg_cgbp = CGBP(model, target_layer, False, use_raw_output_for_gradient, q_percentage)

    def generate_hm(self, inp_tensor, target_class):
        pos_hm = self.pos_cgbp.generate_hm(inp_tensor, target_class)
        neg_hm = self.neg_cgbp.generate_hm(inp_tensor, target_class)

        hm = pos_hm * 1.5 - neg_hm
        hm[hm < 0] = 0
        # plt.subplot(experiment1, 3, experiment1)
        # plt.imshow(pos_hm)
        # plt.subplot(experiment1, 3, 2)
        # plt.imshow(neg_hm)
        # plt.subplot(experiment1, 3, 3)
        # plt.imshow(hm)
        # plt.show()
        # hm = np.where(pos_hm > neg_hm, pos_hm, 0)
        # hm = np.where(pos_hm > neg_hm, pos_hm, pos_hm - neg_hm)
        hm = min_max_normalize(hm)

        # plt.subplot(experiment1, 3, experiment1)
        # plt.imshow(pos_hm)
        # plt.subplot(experiment1, 3, 2)
        # plt.imshow(neg_hm)
        # plt.subplot(experiment1, 3, 3)
        # plt.imshow(hm, cmap='bwr')
        # plt.show()
        return hm

    def generate_bi_hm(self, inp_tensor, target_class):
        pos_hm = self.pos_cgbp.generate_hm(inp_tensor, target_class)
        neg_hm = self.neg_cgbp.generate_hm(inp_tensor, target_class)

        hm = np.where(pos_hm > neg_hm, pos_hm, -neg_hm)

        vmin = hm.min()
        vmax = hm.max()
        hm = (hm - vmin) / (vmax - vmin)
        # plt.subplot(experiment1, 3, experiment1)
        # plt.imshow(pos_hm)
        # plt.subplot(experiment1, 3, 2)
        # plt.imshow(neg_hm)
        # plt.subplot(experiment1, 3, 3)
        # plt.imshow(hm)
        # plt.show()
        # hm = min_max_normalize(hm)
        # hm = np.where(pos_hm > neg_hm, pos_hm, 0)
        # hm = np.where(pos_hm > neg_hm, pos_hm, pos_hm - neg_hm)
        # hm = min_max_normalize(hm)

        return hm