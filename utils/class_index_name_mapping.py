import os.path
from utils import config


def load_index_class_mapping():
    index2class, class2index = {}, {}
    root_data_folder = config['root_data_folder']
    label_label_mapping_path = os.path.join(root_data_folder, config['imagenet_label_mapping'])

    with open(label_label_mapping_path) as file:
        for line in file.readlines():
            idx, cls = line.strip().split(': ')
            idx = int(idx)
            index2class[idx] = cls
            class2index[cls] = idx
    return index2class, class2index


def load_name_class_mapping():
    name2cls, cls2name = {}, {}
    root_data_folder = config['root_data_folder']
    LOC_synset_mapping_path = os.path.join(root_data_folder, config['LOC_synset_mapping'])

    with open(LOC_synset_mapping_path) as file:
        for line in file.readlines():
            name, classes = line.strip().split(' ', maxsplit=1)
            classes = classes.split(', ')
            name2cls[name] = classes
            for cls in classes:
                cls2name[cls] = name
    return name2cls, cls2name


index2class, class2index = load_index_class_mapping()
name2class, class2name = load_name_class_mapping()

if __name__ == '__main__':
    from icecream import ic
    ic(name2class)