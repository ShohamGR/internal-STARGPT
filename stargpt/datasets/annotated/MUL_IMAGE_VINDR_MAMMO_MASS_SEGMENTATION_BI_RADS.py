import os
from os.path import exists, join

from pandas import DataFrame, Series

from tabstar2.utils.datasets import load_csv
from tabstar_paper.datasets.curation_objects import CuratedTarget, CuratedFeature
from tabstar_paper.datasets.objects import SupervisedTask, FeatureType


'''
Dataset Name: huuthocs/vindr-mammo-mammo-mass-segmentation-dataset/
====
Examples: 1488
====
URL: https://www.kaggle.com/huuthocs/vindr-mammo-mammo-mass-segmentation-dataset
====
Description:
title: MVinDr - Mammo Mass Segmentation Dataset
subtitle:
keywords: ['earth and nature']
licenses: [{'name': 'CC-BY-SA-4.0'}]
description: Dataset: VinDr-Mammo
Source: https://vindr.ai/datasets/mammo
Created by: Nguyễn Hữu Thọ (Major Computer Science, Faculty of Information Technology, Vinh Long University of Technology Education)
====
Target Variable: breast_birads (object, 3 distinct): ['BI-RADS 4', 'BI-RADS 3', 'BI-RADS 5']
====
Features:

laterality (object, 2 distinct): ['L', 'R']
view_position (object, 2 distinct): ['MLO', 'CC']
height (int64, 3 distinct): ['3518', '3580', '2812']
width (int64, 25 distinct): ['2800', '2812', '2012', '2609', '2706', '2718', '2702', '2606', '2654', '2638']
breast_density (object, 4 distinct): ['DENSITY C', 'DENSITY B', 'DENSITY D', 'DENSITY A']
finding_categories (object, 26 distinct): ["['Mass']", "['Suspicious Calcification']", "['Suspicious Calcification', 'Mass']", "['Suspicious Lymph Node']", "['Skin Thickening']", "['Focal Asymmetry']", "['Architectural Distortion']", "['Nipple Retraction']", "['Suspicious Calcification', 'Focal Asymmetry']", "['Skin Thickening', 'Nipple Retraction']"]
finding_birads (object, 3 distinct, 6.1% missing): ['BI-RADS 4', 'BI-RADS 3', 'BI-RADS 5']
xmin (float64, 1483 distinct): ['2319.4399', '142.899', '2318.8701', '2063.1299', '12.0563', '-5.1627', '393.11', '431.708', '-5.4943', '2359.32']
ymin (float64, 1482 distinct): ['1507.59', '1200.61', '1331.28', '1545.52', '1166.52', '1339.47', '2104.2618', '2429.7408', '1347.77', '2039.1']
xmax (float64, 1482 distinct): ['2785.3', '2698.9199', '2457.04', '512.321', '2006.0212', '2435.05', '988.06', '977.13', '717.406', '2536.4794']
ymax (float64, 1479 distinct): ['1505.9', '1849.5601', '2097.0', '2034.6899', '1761.3', '1684.03', '1759.0', '1403.9301', '2044.5', '2447.6299']
breast_mass_img (object, 1107 distinct): ['6d4cd11574ad3598cca9b228bcfcc024_8aaea8ad26744fa23f786699ae38d66e.png', '60279474726275dcc024f3c67b4d51a3_712150b595e1a4e62c2e344efd56af42.png', '7ad3e804b8985abc9b72ee0bd4c4a8a3_952db864c34b4f02eeabc1559fda2058.png', '877bde1cb8b7df49cde3fdb7020a1536_733a9b17716849f4fac702227c0414fa.png', '0f1a7b0d4efecb388a4b98ae5e2d5c29_467ec6ad950bb90b6f797d52ac3ab843.png', '8dd986e8174fc84c2a984e66a416ff9f_6a8ea59036175422f3f8253b21b3230e.png', '8b202f831f93639ebcb9cccccec03b09_b8a5dccad0d098308ff3296000716b52.png', '58a2640209d9244ab1eca8f0686effd4_fde1086cce41c66b9efc68270f537837.png', 'a187b425b23c4264a79785c062fce873_9076323f4b661b4b57a9229b869c62c7.png', 'b8acac150be7f949f78dc631a3851f8b_bdf1539e07e60cfcb5e7833f5b63fa86.png']
'''

IMG = "breast_mass_img"

def load_df(dir_path: str) -> DataFrame:
    df = load_csv(dir_path, "finding_annotations.csv")
    df[IMG] = df.apply(lambda row: get_img(row, dir_path=dir_path), axis=1)
    df = df[df[IMG].notna()]
    return df


def get_img(r: Series, dir_path: str) -> str | None:
    img_name = f"{r['study_id']}_{r['image_id']}.png"
    img_folder = join(dir_path, IMAGE_FOLDER)
    full_path = join(img_folder, img_name)
    if not exists(full_path):
        return None
    return img_name

CONTEXT = ""
TARGET = CuratedTarget(raw_name='breast_birads', task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = ['image_id', 'study_id', 'series_id', 'split']
FEATURES = [CuratedFeature(raw_name=IMG, feat_type=FeatureType.IMAGE)]
IMAGE_FOLDER = "images"
LOADING_FUNC = load_df
