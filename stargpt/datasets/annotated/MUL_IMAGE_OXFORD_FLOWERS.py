import os
from os.path import join

from pandas import DataFrame

from tabstar_paper.datasets.curation_objects import CuratedTarget, CuratedFeature
from tabstar_paper.datasets.objects import SupervisedTask, FeatureType
from tabstar_paper.utils.io_handlers import load_json

LABEL_NAME = "Flower Name"
IMAGE_FEATURE_NAME = "Flower"

'''
Dataset Name: waseemalastal/the-oxford-flowers-102-dataset/
====
Examples: 7370
====
URL: https://www.kaggle.com/waseemalastal/the-oxford-flowers-102-dataset
====
Description: 

About Dataset
The Oxford Flowers 102 dataset is a fine-grained classification dataset created by the Visual Geometry Group at the University of Oxford, specifically designed to challenge and test image classification models. This dataset contains 8,189 images of 102 flower species, with each class representing a unique species ranging from common varieties like sunflowers and roses to more exotic blooms.

Dataset Context and Sources
Source: The dataset was developed by the University of Oxford for research on computer vision and machine learning, providing a well-labeled collection of flower images. The dataset's variable conditions (lighting, background, orientation) make it ideal for evaluating models' robustness in real-world scenarios.
Purpose: The goal is to offer a resource for training and evaluating image classification algorithms, especially in fine-grained classification, where the visual differences between categories are subtle. This has made it a popular choice for experimenting with transfer learning models like VGG16, ResNet, and EfficientNet.
Why This Dataset?
With an uneven distribution of images per class, the Oxford Flowers 102 dataset presents a unique challenge for machine learning practitioners aiming to optimize classification accuracy across categories with varying representation. Its high-quality images and multi-class structure have made it a go-to dataset for benchmarking model performance in image classification and fine-tuning pre-trained networks.

This dataset is ideal for researchers, students, and data scientists looking to build and test models for:

Fine-grained classification of natural objects
Transfer learning and model fine-tuning
Evaluating model performance on imbalanced data
====
Target Variable: Flower Name (object, 102 distinct): ['petunia', 'passion flower', 'wallflower', 'watercress', 'water lily', 'rose', 'frangipani', 'foxglove', 'cyclamen', 'lotus']
====
Features:

Flower (object, 7370 distinct): ['train/1/image_06747.jpg', 'train/1/image_06748.jpg', 'train/1/image_06766.jpg', 'train/1/image_06740.jpg', 'train/1/image_06736.jpg', 'train/1/image_06744.jpg', 'train/1/image_06745.jpg', 'train/1/image_06746.jpg', 'train/1/image_06757.jpg', 'train/1/image_06772.jpg']
'''

def load_df(dir_path: str) -> DataFrame:
    df = load_oxford_flowers_df(dir_path)
    df = take_top_10_flower_classes_and_group_others(df)
    return df


def load_oxford_flowers_df(dir_path: str) -> DataFrame:
    ret = []
    image_path_prefix = join(dir_path, IMAGE_FOLDER)
    label_path = join(image_path_prefix, "cat_to_name.json")
    labels = load_json(label_path)
    for split in ["train", "valid"]:
        for n in range(1, 103):
            flower_name = labels[str(n)]
            split_path = f"{split}/{n}"
            for img_name in os.listdir(join(image_path_prefix, split_path)):
                img_path = join(split_path, img_name)
                ret.append({IMAGE_FEATURE_NAME: img_path, LABEL_NAME: flower_name})
    df = DataFrame(ret)
    return df


def take_top_10_flower_classes_and_group_others(df: DataFrame) -> DataFrame:
    top_9_most_common_flower_classes = df[LABEL_NAME].value_counts().head(9).index
    df[LABEL_NAME] = df[LABEL_NAME].apply(lambda x: "other" if x not in top_9_most_common_flower_classes else x)
    return df



CONTEXT = "Object recognition of flower types"
TARGET = CuratedTarget(raw_name=LABEL_NAME, task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name=IMAGE_FEATURE_NAME, feat_type=FeatureType.IMAGE)]
IMAGE_FOLDER = "flower_data/"
LOADING_FUNC = load_df
