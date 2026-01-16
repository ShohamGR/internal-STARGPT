import os
from os.path import exists, join
from typing import Tuple

from pandas import DataFrame

from tabstar2.utils.datasets import load_csv
from tabstar_paper.datasets.curation_objects import CuratedTarget, CuratedFeature
from tabstar_paper.datasets.objects import SupervisedTask, FeatureType


'''
Dataset Name: virenbr11/height-weight-images/
====
Examples: 383
====
URL: https://www.kaggle.com/virenbr11/height-weight-images
====
Description: 
About Dataset
Description
This is the data extracted by me from this website - "https://height-weight-chart.com/"

Contents Downloaded
Images of subject with certain height( '," --> feet and inches ) and weigth('lbs') and a dataset containing the "html link" of image, "src" path of image and height and weight of the subject in the image.
====
Target Variable: weight (float64, 34 distinct): ['95.3', '77.1', '81.6', '104.3', '68.0', '99.8', '108.9', '63.5', '86.2', '72.6']
====
Features:

Filename (object, 383 distinct): ['410-090_Del_L1.jpg', '410-100_Cece_L1.jpg', '410-110_Kaye_L1.jpg', '410-120_Firie_L1.jpg', '410-160_Sarah_L1.jpg', '410-170_Tiffany_L1.jpg', '410-190_TFergusu_L1.jpg', '410-90_Katherine_L1.jpg', '411-100_Beth_L1.jpg', '411-110_Christina_L1.jpg']
height (float64, 7 distinct): ['1.7', '1.8', '1.9', '1.6', '1.5', '2.0', '2.1']
'''

WEIGHT = "weight"
HEIGHT = "height"
RAW_HEIGHT_WEIGHT = "Height & Weight"
FILENAME = "Filename"

def load_df(dir_path: str) -> DataFrame:
    df = load_csv(dir_path, "Output_data.csv")
    assert_image_exists(df, dir_path)
    df[WEIGHT] = df[RAW_HEIGHT_WEIGHT].apply(break_weight)
    df[HEIGHT] = df[RAW_HEIGHT_WEIGHT].apply(break_height)
    return df

def assert_image_exists(df: DataFrame, dir_path: str) -> bool:
    for image_name in df[FILENAME]:
        path = join(dir_path, image_name)
        assert exists(path), f"Image {image_name} not found"


def break_weight(hw: str) -> Tuple[str, str, str]:
    # 4' 10" 100 lbs.
    feet, inches, weight_value = _break_height_weight(hw)
    weight_lbs = float(weight_value)
    LB_TO_KG = 0.45359237
    return round(weight_lbs * LB_TO_KG, 1)


def break_height(hw: str) -> float:
    # 4' 10" 100 lbs.
    feet, inches, weight_value = _break_height_weight(hw)
    assert feet.endswith("'")
    feet = float(feet[:-1])
    assert inches.endswith("\"")
    inches = float(inches[:-1])
    total_inches = feet * 12 + inches
    IN_TO_M = 0.0254
    return round(total_inches * IN_TO_M, 1)


def _break_height_weight(hw: str) -> Tuple[str, str, str]:
    # 4' 10" 100 lbs.
    feet, inches, weight_value, weight_unit = [m.strip() for m in hw.split()]
    assert weight_unit == "lbs."
    return feet, inches, weight_value



CONTEXT = ""
TARGET = CuratedTarget(raw_name=WEIGHT, task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = ['Unnamed: 0', 'Image_link', RAW_HEIGHT_WEIGHT]
FEATURES = [CuratedFeature(raw_name=FILENAME, feat_type=FeatureType.IMAGE)]
IMAGE_FOLDER = ""
LOADING_FUNC = load_df
