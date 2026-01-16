from os.path import exists, join

from pandas import DataFrame

from tabstar2.utils.datasets import load_csv
from tabstar_paper.datasets.curation_objects import CuratedTarget, CuratedFeature
from tabstar_paper.datasets.objects import SupervisedTask, FeatureType


'''
Dataset Name: samoilovmikhail/floral-bouquets-images-and-girlfriend-scores/
====
Examples: 600
====
URL: https://www.kaggle.com/samoilovmikhail/floral-bouquets-images-and-girlfriend-scores
====
Description: 
Floral Bouquets: Images and Girlfriend Scores
Can a model learn my girlfriend's taste in flowers? Images, prices, and ratings.

About Dataset
This dataset contains over 600 images of diverse floral bouquets, each accompanied by its original description, market price, and a subjective human-annotated preference score.

Every bouquet has been manually evaluated by a single annotator and assigned a rating on a scale of 1 (dislike) to 5 (love), reflecting a consistent, personal measure of aesthetic appeal.

This "human-in-the-loop" dataset is perfectly suited for a variety of machine learning tasks, including:

Predictive Modeling: Training a regression or classification model to predict how a person would rate a bouquet based on its image and features.
Personalized Recommendation Systems: Building an engine that can "learn" a user's taste.
Multimodal Analysis: Exploring the relationship between visual features (from images), textual data (from descriptions), and human perception.
====
Target Variable: girlfriend_rating (int64, 5 distinct): ['1', '5', '4', '2', '3']
====
Features:

image_name (object, 600 distinct): ['0000_Bouquet_of_5_eustoms_in_craft.jpg', '0001_Bouquet_of_11_peony-shaped_bush_roses.jpg', '0002_Bouquet_Lydia.jpg', '0003_Bouquet_of_19_white_roses_in_a_designer_package.jpg', '0004_Bouquet_of_15_alstroemeria_with_greenery_in_craft.jpg', '0005_Peonies_9_pieces.jpg', '0006_Sunflowers_9_pieces.jpg', '0007_25_red_roses.jpg', '0008_Red_Roses_Russia_21_pcs.jpg', '0009_Mono_peony-shaped_bush_roses_Madame_Bambastic9_pc.jpg']
description (object, 581 distinct): ['25 red roses', 'French roses', 'A delicate bouquet of bush roses and eustoma combined with pistachios', 'Daisies', 'Bouquet with blue orchid dendrobium combined with pistachio greens', 'Red roses 51 pieces', 'Peony-shaped roses with carnation', 'Peonies are a compliment', 'Bouquet of cosmic orchids dendrobium', 'Bouquet of sunflowers']
rating_by_comments (float64, 37 distinct): ['4.85', '4.89', '4.87', '4.83', '4.9', '4.84', '4.88', '4.92', '4.79', '4.82']
price_rub (int64, 348 distinct): ['4990', '3990', '3500', '4500', '3999', '3850', '3900', '3600', '4900', '3960']
price_usd (float64, 348 distinct): ['62.375', '49.875', '43.75', '56.25', '49.9875', '48.125', '48.75', '45.0', '61.25', '49.5']
'''

IMAGE_FEATURE_NAME = "image_name"

def load_df(dir_path: str) -> DataFrame:
    df = load_csv(dir_path, "flowers.csv")
    assert_image_exists(df, dir_path=dir_path)
    return df


def assert_image_exists(df: DataFrame, dir_path: str):
    image_path = join(dir_path, IMAGE_FOLDER)
    for img in df[IMAGE_FEATURE_NAME]:
        path = join(image_path, img)
        assert exists(path), f"Image {img} not found"


CONTEXT = ""
TARGET = CuratedTarget(raw_name="girlfriend_rating", task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = ['product_id']
FEATURES = [CuratedFeature(raw_name=IMAGE_FEATURE_NAME, feat_type=FeatureType.IMAGE),
            CuratedFeature(raw_name='description', feat_type=FeatureType.TEXT),
]
IMAGE_FOLDER = "images"
LOADING_FUNC = load_df
