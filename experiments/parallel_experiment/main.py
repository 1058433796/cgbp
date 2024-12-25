import os
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager, Value
from torchvision import models
import numpy as np
import torch
from utils import get_dataloader
from explainers import (VanillaGrad, GuidedBackprop, RectGrad, CGBP,IG, IDG, GIG)
from prettytable import PrettyTable

total_running_threads = Value('i', 0)
completed_tasks = Value('i', 0)
def run_explainer_in_process(task, progress_dict, task_id):
    global total_running_threads
    with total_running_threads.get_lock():
        total_running_threads.value += 1
    model_name, name, explainer_func, max_count = task

    # 初始化模型和解释器
    model = getattr(models, model_name)(weights='IMAGENET1K_V1').cuda().eval()
    for param in model.parameters():
        param.requires_grad = False
    explainer = explainer_func(model)
    dataloader = get_dataloader(shuffle=False, num_workers=0)

    explainer_result = {
        'filename': [],
        'label': [],
        'heatmap': []
    }

    count = 0
    for img, label, filename in dataloader:
        img = img.cuda()
        with torch.no_grad():
            output = model(img)
            pred = output.argmax()
            if pred != label[0]:
                continue


        hm = explainer.generate_hm(img, label[0])

        explainer_result['filename'].append(filename[0])
        explainer_result['label'].append(label[0])
        explainer_result['heatmap'].append(hm)

        count += 1
        progress_dict[task_id] = count / max_count
        if count >= max_count:
            break

        del img, label, filename, output, pred, hm

    del model, explainer, dataloader
    torch.cuda.empty_cache()
    # 保存结果
    folder_name = f'./results/{model_name}'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    save_path = f'{folder_name}/{name}.npz'
    np.savez(save_path, **explainer_result)
    print(f'Save {save_path} done')

    with total_running_threads.get_lock():
        total_running_threads.value -= 1
        completed_tasks.value += 1

# 各种解释器生成函数

def create_vanilla_grad(model):
    return VanillaGrad(model)

def create_gbp(model):
    return GuidedBackprop(model)

def create_cgbp(model):
    return CGBP(model)

def create_rect_grad(model):
    return RectGrad(model, q_percentage=0.9)

def create_ig(model):
    return IG(model, n_steps=100)

def create_idg(model):
    return IDG(model, n_steps=100)

def create_gig(model):
    return GIG(model, n_steps=100)


if __name__ == "__main__":
    model_list = [
        'vgg16',
        # 'vgg19',
        'resnet50',
        # 'resnet101'
    ]
    max_count = 400

    explainer_dict = {
        'vanilla_grad': create_vanilla_grad,
        'ig': create_ig,
        'idg': create_idg,
        'gig': create_gig,
        'gbp': create_gbp,
        'rect_grad': create_rect_grad,
        'cgbp': create_cgbp,
    }

    tasks = []
    for model_name in model_list:
        for name, explainer_func in explainer_dict.items():
            tasks.append((model_name, name, explainer_func, max_count))

    with Manager() as manager:
        progress_dict = manager.dict()
        max_workers = 2

        # 创建表格
        table = PrettyTable()
        # 设置表头
        table.field_names = ["Model", "Method", "Progress(%)"]
        start_time = time.time()

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务并等待完成
            for index, task in enumerate(tasks):
                progress_dict[index] = 0.0
                executor.submit(run_explainer_in_process, task, progress_dict, index)

            # 打印进度并覆盖
            while completed_tasks.value < len(tasks):
                table.clear_rows()
                for index in range(len(tasks)):
                    model_name, name, _, _ = tasks[index]
                    table.add_row((model_name, name, f'{progress_dict[index] * 100:.1f}'))

                delta_time = (time.time() - start_time) / 60
                table.add_row(("time", "---", f'{delta_time:.1f} min'))
                table.add_row(("running threads", "---", f'{total_running_threads.value}'))

                print(table)


                time.sleep(60)  # 每60秒更新一次进度



    print("Done")