from os.path import join

from pandas import DataFrame, read_csv

from tabstar_paper.datasets.curation_objects import CuratedTarget, CuratedFeature
from tabstar_paper.datasets.objects import SupervisedTask, FeatureType

LABEL_NAME = "price"
IMAGE_FEATURE_NAME = "watch"

'''
Dataset Name: mathewkouch/a-dataset-of-watches/
====
Examples: 2553
====
URL: https://www.kaggle.com/mathewkouch/a-dataset-of-watches
====
Description:
About Dataset
Intro
This dataset contains 2553 watch images with accompanied product name, brand, and prices.
Images are in .jpg format and contained in the image folder.
metadata.csv file contains the product information.

What you can do with this data
Perform unsupervised learning to cluster and classify watches by styles/colour/gender/material
Train a regression model to predict sale price of watches given its image and brand name
Train generative model to produce novel watch designs

====
Target Variable: price (float64, 307 distinct): ['299.0', '199.0', '249.0', '99.95', '189.0', '149.0', '169.0', '139.0', '289.0', '499.0']
====
Features:

brand (object, 70 distinct): ['Tissot', 'Daniel Wellington', 'Nixon', 'Jag', 'Swatch', 'Guess', 'Fossil', 'Mido', 'Police', 'TONY+WILL']
name (object, 1295 distinct): ['Classic', 'Series 06 Smart Watch', 'Minuit Mesh', 'Silicone Sports Band – The Noosa – Apple Compatible', 'Small Classic', 'Small Astral', 'Lunar', 'Corporal SS Watch', 'Apple Watch Strap - Small', 'Seastar 1000 Chronograph']
watch (object, 2553 distinct): ['0.jpg', '1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg', '6.jpg', '7.jpg', '8.jpg', '9.jpg']
'''

def usd_currency_to_float(usd_str: str) -> float:
    return float(usd_str.replace('$', '').replace(',', '').strip())


def load_df(dir_path: str) -> DataFrame:
    data_path = join(dir_path, "watches/watches/metadata.csv")
    df = read_csv(data_path)
    df.rename(columns={'image_name': IMAGE_FEATURE_NAME}, inplace=True)
    df.drop(columns=['Unnamed: 0'], inplace=True)
    df[LABEL_NAME] = df[LABEL_NAME].apply(usd_currency_to_float)
    return df


CONTEXT = "Branded Watches Prices"
TARGET = CuratedTarget(raw_name=LABEL_NAME, task_type=SupervisedTask.REGRESSION)
FEATURES = [CuratedFeature(raw_name=IMAGE_FEATURE_NAME, feat_type=FeatureType.IMAGE),
            CuratedFeature(raw_name="name", feat_type=FeatureType.TEXT)]
IMAGE_FOLDER = "watches/watches/images"
LOADING_FUNC = load_df
