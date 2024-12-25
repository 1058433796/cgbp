from concurrent.futures import ProcessPoolExecutor

import numpy as np
import os
import glob
from multiprocessing import Manager, Value
from prettytable import PrettyTable
from torchvision import models
from evaluation.evaluator import Evaluator
import torch
from utils.image_process import preprocess_image
import time

total_running_threads = Value('i', 0)
completed_tasks = Value('i', 0)
def load_npz_files(npz_files_path, model_name):
    return glob.glob(os.path.join(npz_files_path, model_name, '*.npz'))

def evaluate_explainer(task, progress_dict, task_id):
    global total_running_threads
    with total_running_threads.get_lock():
        total_running_threads.value += 1
    model_name, npz_file = task
    model = getattr(models, model_name)(weights='IMAGENET1K_V1').cuda().eval()
    for param in model.parameters():
        param.requires_grad = False
    evaluator = Evaluator(model)

    explainer_result = np.load(npz_file)
    explainer_name = os.path.basename(npz_file).split('.')[0]

    filenames = explainer_result['filename']
    labels = explainer_result['label']
    heatmaps = explainer_result['heatmap']

    sic_scores = []
    aic_scores = []
    deletion_scores = []
    insertion_scores = []

    total_count = len(filenames)
    for idx, (filename, label, heatmap) in enumerate(zip(filenames, labels, heatmaps)):
        if np.isnan(heatmap).any():
            print(filename, label, 'skipped')
            continue

        img_tensor = preprocess_image(filename)
        img_nd = img_tensor.numpy()

        sic_results = None
        aic_results = None
        try:
            sic_results = evaluator.sic_score(img_nd, heatmap, label)
            aic_results = evaluator.aic_score(img_nd, heatmap, label)
        except:
            pass

        deletion_score = evaluator.deletion_score(img_tensor, heatmap)
        insertion_score = evaluator.insertion_score(img_tensor, heatmap)
        print(insertion_score, deletion_score)
        del img_tensor, img_nd, heatmap

        if sic_results is not None:
            sic_scores.append(sic_results.auc)
        if aic_results is not None:
            aic_scores.append(aic_results.auc)

        if deletion_score is not None:
            deletion_scores.append(deletion_score)

        if insertion_score is not None:
            insertion_scores.append(insertion_score)

        progress_dict[task_id] = (idx + 1) / total_count

    del model, evaluator
    torch.cuda.empty_cache()
    aggregate_scores(sic_scores, aic_scores, deletion_scores, insertion_scores, model_name, explainer_name)

    with total_running_threads.get_lock():
        total_running_threads.value -= 1
        completed_tasks.value += 1

def aggregate_scores(sic_scores, aic_scores, deletion_scores, insertion_scores, model_name, explainer_name):
    sic_score = np.mean(sic_scores) if sic_scores else 0
    aic_score = np.mean(aic_scores) if aic_scores else 0
    deletion_score = np.mean(deletion_scores) if deletion_scores else 0
    insertion_score = np.mean(insertion_scores) if insertion_scores else 0

    evaluation_result = {
        'sic_score': sic_score,
        'aic_score': aic_score,
        'deletion_score': deletion_score,
        'insertion_score': insertion_score,
        'ins_sub_del_score': insertion_score - deletion_score,
    }
    save_dir = f'./results/{model_name}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = f'{save_dir}/{explainer_name}.npz'
    np.savez(save_path, **evaluation_result)


if __name__ == "__main__":
    npz_files_path = '../experiments/parallel_experiment/results'
    model_list = [
        'vgg16',
        # 'vgg19',
        # 'resnet50',
        # 'resnet101'
    ]
    explainers = [
        'vanilla_grad',
        'gbp',
        'rect_grad',
        'cgbp',
        'ig',
        'gig',
        'idg',
    ]
    tasks = []
    for model_name in model_list:
        npz_files = load_npz_files(npz_files_path, model_name)
        for npz_file in npz_files:
            print(npz_file)
            explainer_name = os.path.basename(npz_file).split('.')[0]
            if explainer_name not in explainers:
                continue
            tasks.append((model_name, npz_file))

    with Manager() as manager:
        progress_dict = manager.dict()
        max_workers = 1

        # 创建表格
        table = PrettyTable()
        # 设置表头
        table.field_names = ["Model", "Method", "Progress(%)"]
        start_time = time.time()

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务并等待完成
            for index, task in enumerate(tasks):
                progress_dict[index] = 0.0
                executor.submit(evaluate_explainer, task, progress_dict, index)

            # 打印进度并覆盖
            while completed_tasks.value < len(tasks):
                table.clear_rows()
                for index in range(len(tasks)):
                    model_name, npz_file = tasks[index]
                    explainer_name = os.path.basename(npz_file).split('.')[0]
                    table.add_row((model_name, explainer_name, f'{progress_dict[index] * 100:.1f}'))

                delta_time = (time.time() - start_time) / 60
                table.add_row(("time", "---", f'{delta_time:.1f} min'))
                table.add_row(("running threads", "---", f'{total_running_threads.value}'))
                print(table)

                time.sleep(60)  # 每60秒更新一次进度


