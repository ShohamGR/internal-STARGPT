import os
from os.path import exists, join
from typing import Optional

from pandas import DataFrame, read_excel

from tabstar_paper.datasets.curation_objects import CuratedTarget, CuratedFeature
from tabstar_paper.datasets.objects import SupervisedTask, FeatureType


'''
Dataset Name: mypapit/mangomassnet552-dataset/
====
Examples: 546
====
URL: https://www.kaggle.com/mypapit/mangomassnet552-dataset
====
Description: 
MangoMassNet-552 Dataset
Mango Fruit Mass Estimation Dataset (Harumanis)

About Dataset
Context
The dataset was created because there was no freely available Mango image dataset which includes mango images, its grade and mass/weight.

This dataset contains 552 images of Harumanis Mango (clone number MA 128) collected from Fruit Collection Center, FAMA Perlis, Malaysia. The images in the dataset is resize according to A4 paper ratio of 8:10. All mango samples are taken on top of blank A4 paper, as the paper is used as visual cue for mass estimation.

You can refer to the paper found in IEEEXplore titled "Mango Mass Estimation from RGB Image with Convolutional Neural Network" for more information.

Citation
If you use the dataset, please cite this dataset and the paper:

Paper (IEEE)
M. H. Ismail, M. N. Wagimin and T. R. Razak, "Estimating Mango Mass from RGB Image with Convolutional Neural Network," 2022 3rd International Conference on Artificial Intelligence and Data Sciences (AiDAS), IPOH, Malaysia, 2022, pp. 105-110, doi: 10.1109/AiDAS56890.2022.9918807.

Dataset
Mohammad Hafiz bin Ismail, & Mohd Nazuan Wagimin. (2022). MangoMassNet-552 Dataset [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/3987156

Inspiration
Mass estimation of mangoes / Regression problems
Automated mango grading
Image classification of mangoes
Image classification of Mango variety (cv Harumanis)
License and Copyright
Copyright 2022 (c) Mohammad Hafiz bin Ismail, Mohd Nazuan Wagimin

This dataset is licensed under Attribution-ShareAlike 4.0 International (CC BY-SA 4.0).
You can use the dataset in your work, research, study, etc. Provided that you cite/attribute the dataset properly
====
Target Variable: Mass(kg) (float64, 36 distinct): ['0.45', '0.5', '0.4', '0.55', '0.35', '0.3', '0.6', '0.41', '0.53', '0.54']
====
Features:

Fruit No (object, 546 distinct): ['1a.jpg', '2a.jpg', '3a.jpg', '4a.jpg', '5a.jpg', '6a.jpg', '7a.jpg', '8a.jpg', '9a.jpg', '10a.jpg']
Color_K-Yellow_P_Green (object, 2 distinct): ['P', 'K']
Fruit Grade (object, 3 distinct): ['1', '2', 'P']
'''

def load_df(dir_path: str) -> DataFrame:
    df = read_excel(join(dir_path, "Harumanis_mango_weight_grade.xlsx"))
    df[IMG] = df[IMG].apply(lambda img: remove_missing_image(img, dir_path))
    df = df[df[IMG].notna()]
    return df


def remove_missing_image(img: str, dir_path: str) -> Optional[str]:
    if not exists(join(dir_path, IMAGE_FOLDER, img)):
        return None
    return img
        


IMG = "Fruit No"


CONTEXT = ""
TARGET = CuratedTarget(raw_name='Mass(kg)', task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name=IMG, feat_type=FeatureType.IMAGE)]
IMAGE_FOLDER = "images"
LOADING_FUNC = load_df
