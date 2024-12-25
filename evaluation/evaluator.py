import copy
import numpy as np
import torch
from evaluation.util.pic_functions import compute_pic_metric, generate_random_mask
from utils.image_process import tensor_to_image
from torchvision import transforms
from utils.config_loader import config
from .util import RISETestFunctions as RISE

class Evaluator(object):

    def __init__(self, model, img_hw=224):
        self.model = model
        self.model.eval()
        self.random_mask = None
        deletion, insertion = self._get_deletion_and_insertion_func(img_hw)
        self.deletion_metric = deletion
        self.insertion_metric = insertion

    def _get_deletion_and_insertion_func(self, img_hw):
        klen = 11
        ksig = 5
        kern = RISE.gkern(klen, ksig)
        # img_hw = 224
        blur = lambda x: torch.nn.functional.conv2d(x, kern.to(x.device), padding=klen // 2)
        insertion = RISE.CausalMetric(self.model, img_hw * img_hw, 'ins', img_hw, substrate_fn=blur)
        deletion = RISE.CausalMetric(self.model, img_hw * img_hw, 'del', img_hw, substrate_fn=torch.zeros_like)
        return deletion, insertion

    def deletion_score(self, input_image, input_exp, target_class=None):
        assert isinstance(input_image, torch.Tensor)
        if input_image.dim() == 3:
            input_image = input_image.unsqueeze(0)
        steps, arr = self.deletion_metric.single_run(input_image, input_exp, 'cuda', target_class)
        auc = RISE.auc(arr)
        return auc

    def insertion_score(self, input_image, input_exp, target_class=None):
        assert isinstance(input_image, torch.Tensor)
        if input_image.dim() == 3:
            input_image = input_image.unsqueeze(0)
        steps, arr = self.insertion_metric.single_run(input_image, input_exp, 'cuda', target_class)
        auc = RISE.auc(arr)
        return auc


    def pointing_game(self, heatmap, bndboxs):
        '''
        :param heatmap: numpy array like: H * W
        :param bndboxs: list of bndbox like: [(xmin, ymin, xmax, ymax), ...]
        '''
        if len(bndboxs) == 0: return 1.0
        heatmap = copy.deepcopy(heatmap)
        box_mask = np.zeros_like(heatmap)
        for (xmin, ymin, xmax, ymax) in bndboxs:
            box_mask[ymin:ymax + 1, xmin: xmax + 1] = 1
        energy_in_box = (heatmap * box_mask).sum()
        energy_out_of_box = (heatmap * (1 - box_mask)).sum()
        proportion = energy_in_box / (energy_in_box + energy_out_of_box)
        return proportion

    def get_soft_prediction_func(self, target_class):
        transform = transforms.Compose([
            transforms.ToTensor(),  # 转换为[0,experiment1]的张量
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
        ])
        def predict(image_batch):
            "将B * H * W * C的uint8图像作为输入 预测目标类别的结果"
            img_as_tensor_list = [transform(img) for img in image_batch]
            img_as_tensor = torch.stack(img_as_tensor_list, dim=0)
            img_as_tensor = img_as_tensor.cuda()
            with torch.no_grad():
                output = self.model(img_as_tensor)
                prob = torch.softmax(output, 1)
            return prob[:, target_class].detach().cpu().numpy()

        return predict

    def get_acc_prediction_func(self, target_class):
        transform = transforms.Compose([
            transforms.ToTensor(),  # 转换为[0,experiment1]的张量
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
        ])
        def predict(image_batch):
            "将B * H * W * C的uint8图像作为输入 预测目标类别的结果"
            # img_batch = np.transpose(image_batch, (0, 3, experiment1, 2))
            img_as_tensor_list = [transform(img) for img in image_batch]
            img_as_tensor = torch.stack(img_as_tensor_list, dim=0)
            img_as_tensor = img_as_tensor.cuda()
            with torch.no_grad():
                output = self.model(img_as_tensor)
                arg_max = output.argmax(1)
                accuracy = torch.where(arg_max == target_class, 1., 0.)
            return accuracy.detach().cpu().numpy()

        return predict

    def sic_score(self, input_image, input_exp, target_class):
        """

        :param input_image: numpy array like: C * H * W
        :param input_exp: numpy array like:H * W
        :param target_class: int
        :return:
        """
        # img 转为uint8类型 shape为 H * W * C
        input_image = tensor_to_image(input_image)
        input_image = np.array(input_image)

        # pred_func 能够使用转换后的img作为输入进行预测的func
        pred_func = self.get_soft_prediction_func(target_class)
        if self.random_mask is None:
            img_size = int(config['crop_size'])
            self.random_mask = generate_random_mask(img_size, img_size)

        saliency_thresholds = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.13, 0.21, 0.34, 0.5, 0.75]
        sic_results = compute_pic_metric(
            img=input_image,
            saliency_map=input_exp,
            random_mask=self.random_mask,
            pred_func=pred_func,
            saliency_thresholds=saliency_thresholds,
            min_pred_value=0.5
        )
        return sic_results

    def aic_score(self, input_image, input_exp, target_class):
        # img 转为uint8类型 shape为 H * W * C
        input_image = tensor_to_image(input_image)
        input_image = np.array(input_image)

        # pred_func 能够使用转换后的img作为输入进行预测的func
        pred_func = self.get_acc_prediction_func(target_class)
        if self.random_mask is None:
            img_size = int(config['crop_size'])
            self.random_mask = generate_random_mask(img_size, img_size)

        saliency_thresholds = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.13, 0.21, 0.34, 0.5, 0.75]

        aic_results = compute_pic_metric(
            img=input_image,
            saliency_map=input_exp,
            random_mask=self.random_mask,
            pred_func=pred_func,
            saliency_thresholds=saliency_thresholds,
            min_pred_value=0.5
        )
        return aic_results


