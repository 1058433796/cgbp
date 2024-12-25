import numpy as np
import os, glob
from prettytable import PrettyTable

def plot_results(model_name, explainer_list, npz_files_path, npz_files):
    sic_scores = []
    aic_scores = []
    deletion_scores = []
    insertion_scores = []
    ins_sub_del_scores = []

    # table = PrettyTable(["Metric", "Model", "Grad", "IG", "IDG", "GIG", "GBP", "RectGrad", "CGBP"])
    valid_explainer_list = []
    for explainer_name in explainer_list:
        npz_file = os.path.join(npz_files_path, model_name, explainer_name + '.npz')
        print(npz_file)
        if npz_file not in npz_files: continue

        evaluation_result = np.load(npz_file)
        sic_score = evaluation_result['sic_score']
        aic_score = evaluation_result['aic_score']
        deletion_score = evaluation_result['deletion_score']
        insertion_score = evaluation_result['insertion_score']
        ins_sub_del_score = evaluation_result['ins_sub_del_score']

        sic_scores.append(f'{sic_score:.3f}')
        aic_scores.append(f'{aic_score:.3f}')
        deletion_scores.append(f'{deletion_score:.3f}')
        insertion_scores.append(f'{insertion_score:.3f}')
        ins_sub_del_scores.append(f'{ins_sub_del_score:.3f}')
        valid_explainer_list.append(explainer_name)

    table.add_row(("SIC", model_name, *sic_scores))
    table.add_row(("AIC", model_name, *aic_scores))
    table.add_row(("Ins", model_name, *insertion_scores))
    table.add_row(("Del", model_name, *deletion_scores))
    print(valid_explainer_list)

    print(table)



if __name__ == '__main__':
    npz_files_path = '../evaluation/results'
    model_list = [
        'vgg16',
        # 'vgg19',
        # 'resnet50',
        # 'resnet101'
    ]
    explainers = [
        # 'vanilla_grad',
        # 'ig',
        # 'idg',
        # 'gig',
        # 'gbp',
        # 'rect_grad',
        # 'cgbp',
    ]
    for model_name in model_list:
        npz_files = glob.glob(os.path.join(npz_files_path, model_name, '*.npz'))
        plot_results(model_name, explainers, npz_files_path, npz_files)