import torch
import numpy as np
from utils.image_process import min_max_normalize
from .util.attribution_methods import saliencyMethods as attribution
from .util.attribution_methods import GIGBuilder as GIG_Builder


class IG:
    def __init__(self, model, n_steps=100, batch_size=50):
        self.kwargs = {
            'model': model,
            'steps': n_steps,
            'batch_size': batch_size,
            'alpha_star': 1,
            'baseline': 0,
            'device': "cuda" if torch.cuda.is_available() else "cpu",
        }

    def generate_hm(self, inp_tensor, target_class):
        grad = attribution.IG(inp_tensor, target_class=target_class, **self.kwargs)
        hm = grad.squeeze().sum(0).abs()
        # hm = grad.squeeze().abs().sum(0)
        hm = min_max_normalize(hm)
        return hm.detach().cpu().numpy()


class LIG:
    def __init__(self, model, n_steps=100, batch_size=50):
        self.kwargs = {
            'model': model,
            'steps': n_steps,
            'batch_size': batch_size,
            'alpha_star': 0.9,
            'baseline': 0,
            'device': "cuda" if torch.cuda.is_available() else "cpu",
        }

    def generate_hm(self, inp_tensor, target_class):
        grad = attribution.IG(inp_tensor, target_class=target_class, **self.kwargs)
        hm = grad.squeeze().abs().sum(0)
        hm = min_max_normalize(hm)
        return hm.detach().cpu().numpy()


class IDG:
    def __init__(self, model, n_steps=100, batch_size=50):
        self.kwargs = {
            'model': model,
            'steps': n_steps,
            'batch_size': batch_size,
            'baseline': 0,
            'device': "cuda" if torch.cuda.is_available() else "cpu",
        }

    def generate_hm(self, inp_tensor, target_class):
        grad = attribution.IDG(inp_tensor, target_class=target_class, **self.kwargs)
        hm = grad.squeeze().sum(0).abs()
        # hm = grad.squeeze().abs().sum(0)
        hm = min_max_normalize(hm)
        return hm.detach().cpu().numpy()


class GIG:
    def __init__(self, model, n_steps=100):
        self.guided_ig = GIG_Builder.GuidedIG()
        self.kwargs = {
            'model': model,
            'device': "cuda" if torch.cuda.is_available() else "cpu",
            'call_model_function': GIG_Builder.call_model_function,
            'x_steps': n_steps,
        }

    def generate_hm(self, inp_tensor, target_class):
        call_model_args = {'class_idx_str': target_class}
        grad = self.guided_ig.GetMask(inp_tensor, call_model_args=call_model_args, **self.kwargs)
        hm = grad.squeeze().sum(0).abs()
        # hm = grad.squeeze().abs().sum(0)
        hm = min_max_normalize(hm)
        return hm.detach().cpu().numpy()
