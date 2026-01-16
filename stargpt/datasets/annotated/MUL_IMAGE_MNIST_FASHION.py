import os
from os.path import join

import pandas as pd
from pandas import DataFrame

from tabstar_paper.datasets.curation_objects import CuratedFeature, CuratedTarget
from tabstar_paper.datasets.objects import SupervisedTask, FeatureType

'''
Dataset Name: andhikawb/fashion-mnist-png/
====
Examples: 70000
====
URL: https://www.kaggle.com/andhikawb/fashion-mnist-png
====
Description: 

About Dataset
PNG version of Fashion MNIST (with correct labels), taken from GitHub (thanks to gazay/Alex Gaziev). Contains 70,000 28x28 images (60,000 train and 10,000 test). See original dataset here (Kaggle).

Label	Description
0	T-shirt/top
1	Trouser
2	Pullover
3	Dress
4	Coat
5	Sandal
6	Shirt
7	Sneaker
8	Bag
9	Ankle boot

OpenML 40996:
Description: **Author**: Han Xiao, Kashif Rasul, Roland Vollgraf  
**Source**: [Zalando Research](https://github.com/zalandoresearch/fashion-mnist)  
**Please cite**: Han Xiao and Kashif Rasul and Roland Vollgraf, Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms, arXiv, cs.LG/1708.07747  

Fashion-MNIST is a dataset of Zalando's article images, consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. Fashion-MNIST is intended to serve as a direct drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms. It shares the same image size and structure of training and testing splits. 

Raw data available at: https://github.com/zalandoresearch/fashion-mnist

### Target classes
Each training and test example is assigned to one of the following labels:
Label  Description  
0  T-shirt/top  
1  Trouser  
2  Pullover  
3  Dress  
4  Coat  
5  Sandal  
6  Shirt  
7  Sneaker  
8  Bag  
9  Ankle boot
====
Target Variable: Wearing Item (object, 10 distinct): ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
====
Features:

Fashion Item Image (object, 70000 distinct): ['train/0/5268.png', 'train/0/27174.png', 'train/0/36299.png', 'train/0/8384.png', 'train/0/45849.png', 'train/0/55641.png', 'train/0/44561.png', 'train/0/27698.png', 'train/0/5329.png', 'train/0/1169.png']
'''


LABEL_MAP = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',
             5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}

LABEL_NAME = "Wearing Item"
IMAGE_FEATURE_NAME = "Fashion Item Image"


def load_df(dir_path: str) -> DataFrame:
    ret = []
    for split in ["train", "test"]:
        for n in range(10):
            label_name = LABEL_MAP[n]
            img_dir = join(split, str(n))
            for img in os.listdir(join(dir_path, img_dir)):
                img_path = join(img_dir, img)
                ret.append({IMAGE_FEATURE_NAME: img_path, LABEL_NAME: label_name})
    df = pd.DataFrame(ret)
    return df


CONTEXT = "Image Object Classification - MNIST Fashion"
TARGET = CuratedTarget(raw_name=LABEL_NAME, task_type=SupervisedTask.MULTICLASS)
FEATURES = [CuratedFeature(raw_name=IMAGE_FEATURE_NAME, feat_type=FeatureType.IMAGE)]
IMAGE_FOLDER = ""
LOADING_FUNC = load_df
