import os
from os.path import join

from pandas import DataFrame

from tabstar2.utils.datasets import load_csv
from tabstar_paper.datasets.curation_objects import CuratedTarget, CuratedFeature
from tabstar_paper.datasets.objects import SupervisedTask, FeatureType


'''
Dataset Name: ted8080/house-prices-and-images-socal/
====
Examples: 15474
====
URL: https://www.kaggle.com/ted8080/house-prices-and-images-socal
====
Description: 
About Dataset
Context
I created this dataset to predict the house price from its image(s). It has the price and corresponding image. Each house has only one image.

Content
The data contains 7 columns and over 15000 rows.
Image_id refers to the image in the image folder.
n_citi is the label encode of the citi column.

To clean unwanted images, please go to this link.
github.com/ted2020/House-Price-Prediction-via-Computer-Vision

Inspiration
I hope to predict the price of a house from its images. For now, the dataset only includes the exterior images of a house.
====
Target Variable: price (int64, 2320 distinct): ['699000', '799000', '749000', '899000', '599000', '650000', '499000', '649000', '550000', '750000']
====
Features:

image_id (object, 15474 distinct): ['0.jpg', '1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg', '6.jpg', '7.jpg', '8.jpg', '9.jpg']
street (object, 12401 distinct): ['Address not provided', '1930 W San Marcos Blvd', '65565 Acoma Avenue', '650 S Rancho Santa Fe Rd', '315 Verbena Drive', '2055 Bonita Street', '550 Higuera Street', '6665 Mission Gorge Rd', '2846 Griffin Avenue', '4374 Nautilus Way']
citi (object, 415 distinct): ['San Diego, CA', 'Los Angeles, CA', 'Lancaster, CA', 'La Quinta, CA', 'Riverside, CA', 'Corona, CA', 'Escondido, CA', 'Fontana, CA', 'Big Bear, CA', 'Palm Springs, CA']
n_citi (int64, 415 distinct): ['320', '207', '193', '175', '310', '87', '115', '119', '38', '266']
bed (int64, 12 distinct): ['3', '4', '2', '5', '6', '1', '7', '8', '10', '9']
bath (float64, 32 distinct): ['2.0', '3.0', '2.1', '1.0', '3.1', '4.1', '4.0', '1.1', '5.0', '5.1']
sqft (int64, 3571 distinct): ['1200', '1100', '1600', '1440', '1300', '1344', '1000', '1800', '1400', '1500']
'''

def load_df(dir_path: str) -> DataFrame:
    df = load_csv(dir_path, "socal2.csv")
    return df


def _img_to_path(img_id: int) -> str:
    return f"{img_id}.jpg"

CONTEXT = ""
TARGET = CuratedTarget(raw_name="price", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name="image_id", feat_type=FeatureType.IMAGE, processing_func=_img_to_path),
            CuratedFeature(raw_name="street", feat_type=FeatureType.TEXT),
            CuratedFeature(raw_name="citi", feat_type=FeatureType.TEXT),
            ]
IMAGE_FOLDER = "socal2/socal_pics"
LOADING_FUNC = load_df
