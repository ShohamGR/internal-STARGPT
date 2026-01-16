from os.path import join

from pandas import DataFrame
from scipy.io import loadmat

from tabstar_paper.datasets.curation_objects import CuratedTarget, CuratedFeature
from tabstar_paper.datasets.objects import SupervisedTask, FeatureType

LABEL_NAME = "Standford Car"
IMAGE_FEATURE_NAME = "Car Image"

'''
Dataset Name: eduardo4jesus/stanford-cars-dataset/
====
Examples: 8144
====
URL: https://www.kaggle.com/eduardo4jesus/stanford-cars-dataset
====
Description: 
Overview
The Cars dataset contains 16,185 images of 196 classes of cars. The data is split into 8,144 training images and 8,041 testing images, where each class has been split roughly in a 50-50 split. Classes are typically at the level of Make, Model, Year, e.g. 2012 Tesla Model S or 2012 BMW M3 coupe.

Acknowledgements
Data source and banner image:
http://ai.stanford.edu/~jkrause/cars/car_dataset.html contains all bounding boxes and labels for both training and tests.

If you use this dataset, please cite the following paper:

3D Object Representations for Fine-Grained Categorization

Jonathan Krause, Michael Stark, Jia Deng, Li Fei-Fei

4th IEEE Workshop on 3D Representation and Recognition, at ICCV 2013 (3dRR-13). Sydney, Australia. Dec. 8, 2013.

Inspiration
Can you form a model that can tell the difference between cars by
type or colour?

Which cars are manufactured by Tesla vs BMW?
====
Target Variable: Standford Car (object, 196 distinct): ['GMC Savana Van 2012', 'Chrysler 300 SRT-8 2010', 'Mitsubishi Lancer Sedan 2012', 'Mercedes-Benz 300-Class Convertible 1993', 'Jaguar XK XKR 2012', 'Chevrolet Corvette ZR1 2012', 'Volkswagen Golf Hatchback 1991', 'Nissan 240SX Coupe 1998', 'Audi S6 Sedan 2011', 'Bentley Continental GT Coupe 2007']
====
Features:

Car Image (object, 8144 distinct): ['00001.jpg', '00002.jpg', '00003.jpg', '00004.jpg', '00005.jpg', '00006.jpg', '00007.jpg', '00008.jpg', '00009.jpg', '00010.jpg']
'''

def load_df(dir_path: str) -> DataFrame:
    df = load_standford_cars(dir_path)
    df = take_9_most_common_car_classes_and_group_others(df)
    return df

def take_9_most_common_car_classes_and_group_others(df: DataFrame) -> DataFrame:
    top_9_most_common_car_classes = df[LABEL_NAME].value_counts().head(9).index
    df[LABEL_NAME] = df[LABEL_NAME].apply(lambda x: "other" if x not in top_9_most_common_car_classes else x)
    return df

def load_standford_cars(dir_path: str):
    annotation_dir = join(dir_path, 'car_devkit/devkit/')
    labels_mat = loadmat(join(annotation_dir, 'cars_meta.mat'))
    labels = [str(x[0]) for x in labels_mat['class_names'].ravel()]
    train_mat = loadmat(join(annotation_dir, 'cars_train_annos.mat'))
    ret = []
    for example in train_mat['annotations'].flatten():
        img_label = int(example['class'][0][0])
        label = labels[img_label - 1]
        img_fname = str(example['fname'][0])
        ret.append({IMAGE_FEATURE_NAME: img_fname, LABEL_NAME: label})
    ret = DataFrame(ret)
    return ret


CONTEXT = "Object Detection of different types of Stanford Cars"
TARGET = CuratedTarget(raw_name=LABEL_NAME, task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name=IMAGE_FEATURE_NAME, feat_type=FeatureType.IMAGE)]
IMAGE_FOLDER = "cars_train/cars_train"
LOADING_FUNC = load_df
