import os
from os.path import join

from pandas import DataFrame

from tabstar_paper.datasets.curation_objects import CuratedTarget, CuratedFeature
from tabstar_paper.datasets.objects import SupervisedTask, FeatureType

''''
Dataset Name: tanlikesmath/the-oxfordiiit-pet-dataset/
====
Examples: 7390
====
URL: https://www.kaggle.com/tanlikesmath/the-oxfordiiit-pet-dataset
====
Description:
About Dataset
The Oxford-IIIT Pet Dataset is a 37 category pet dataset with roughly 200 images for each class created by the Visual Geometry Group at Oxford. The images have a large variations in scale, pose and lighting. All images have an associated ground truth annotation of breed, head ROI, and pixel level trimap segmentation.
====
Target Variable: Pet Type (object, 37 distinct): ['Russian_Blue', 'american_pit_bull_terrier', 'Maine_Coon', 'Birman', 'saint_bernard', 'basset_hound', 'Persian', 'english_setter', 'leonberger', 'american_bulldog']
====
Features:

Pet (object, 7390 distinct): ['Russian_Blue_158.jpg', 'american_pit_bull_terrier_198.jpg', 'Maine_Coon_154.jpg', 'Birman_174.jpg', 'Birman_33.jpg', 'saint_bernard_165.jpg', 'basset_hound_117.jpg', 'Persian_80.jpg', 'english_setter_194.jpg', 'Maine_Coon_213.jpg']
'''


LABEL_NAME = "Pet Type"
IMAGE_FEATURE_NAME = "Pet"



def load_df(dir_path: str) -> DataFrame:
    df = load_oxford_pets_df(dir_path)
    df = take_top_10_most_similar_pets(df)
    return df

def load_oxford_pets_df(dir_path: str) -> DataFrame:
    ret = []
    images_path = join(dir_path, IMAGE_FOLDER)
    for filename in os.listdir(images_path):
        if not filename.endswith(".jpg"):
            continue
        label = filename.replace('.jpg', '')
        label, idx = label.rsplit("_", 1)
        assert idx.isdigit()
        assert all(c.isalpha() or c == "_" for c in label)
        ret.append({IMAGE_FEATURE_NAME: filename, LABEL_NAME: label})
    df = DataFrame(ret)
    return df

def take_top_10_most_similar_pets(df: DataFrame) -> DataFrame:
    """['Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair', 'Egyptian_Mau', 'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue', 'Siamese', 'Sphynx', 
    'american_bulldog', 'american_pit_bull_terrier', 'basset_hound', 'beagle', 'boxer', 'chihuahua', 'english_cocker_spaniel', 'english_setter', 
    'german_shorthaired', 'great_pyrenees', 'havanese', 'japanese_chin', 'keeshond', 'leonberger', 'miniature_pinscher', 'newfoundland', 'pomeranian', 
    'pug', 'saint_bernard', 'samoyed', 'scottish_terrier', 'shiba_inu', 'staffordshire_bull_terrier', 'wheaten_terrier', 'yorkshire_terrier']"""
    similar_looking_dogs = dogs = [
    "american_pit_bull_terrier",
    "staffordshire_bull_terrier",
    "american_bulldog",
    "pomeranian",
    "yorkshire_terrier",
    "miniature_pinscher",
    "beagle",
    "english_cocker_spaniel",
    "basset_hound",
    "boxer"
]
    df = df[df[LABEL_NAME].isin(similar_looking_dogs)]
    return df


CONTEXT = "Object recognition of different types of pets"
TARGET = CuratedTarget(raw_name=LABEL_NAME, task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name=IMAGE_FEATURE_NAME, feat_type=FeatureType.IMAGE)]
IMAGE_FOLDER = "images"
LOADING_FUNC = load_df
