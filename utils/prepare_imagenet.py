# 将imagenet 处理成imageFolder能接受的形式
# 目前不需要这样做了
import os
import pandas as pd
import shutil

def prepare_imagenet(raw_data_folder, preprocessed_data_folder, csv_file):
    # 读取CSV文件
    df = pd.read_csv(csv_file, names=('filename', 'label'), sep=' ')

    # 遍历CSV中的每一行，并处理图片
    for index, row in df.iterrows():
        # 图像源路径
        src_path = os.path.join(raw_data_folder, row['filename'])
        target_data_folder = os.path.join(preprocessed_data_folder, str(row['label']).zfill(4))
        # 如果目标文件夹不存在，则创建它
        if not os.path.exists(target_data_folder):
            os.makedirs(target_data_folder)

        # 构建目标路径并移动图片
        dst_path = os.path.join(target_data_folder, row['filename'])
        shutil.copy(src_path, dst_path)


# 使用示例
if __name__ == "__main__":
    from utils.config_loader import config
    root_data_folder = config['root_data_folder']
    raw_data_folder = os.path.join(root_data_folder, config['raw_data_folder'])
    preprocessed_data_folder = os.path.join(root_data_folder, 'processed')
    csv_file = os.path.join(root_data_folder, config['ground_truth_csv'])
    prepare_imagenet(raw_data_folder, preprocessed_data_folder, csv_file)
