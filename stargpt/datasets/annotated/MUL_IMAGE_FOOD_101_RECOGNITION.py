import os
from os.path import join

from pandas import DataFrame

from tabstar_paper.datasets.curation_objects import CuratedTarget, CuratedFeature
from tabstar_paper.datasets.objects import SupervisedTask, FeatureType

LABEL_NAME = "food_type"
IMAGE_FEATURE_NAME = "food_image"

''''
Dataset Name: dansbecker/food-101/
====
Examples: 101000
====
URL: https://www.kaggle.com/dansbecker/food-101
====
Description: 
About Dataset
Context
There's a story behind every dataset and here's your opportunity to share yours.

Content
This is the Food 101 dataset, also available from https://www.vision.ee.ethz.ch/datasets_extra/food-101/

It contains images of food, organized by type of food. It was used in the Paper "Food-101 – Mining Discriminative Components with Random Forests" by Lukas Bossard, Matthieu Guillaumin and Luc Van Gool. It's a good (large dataset) for testing computer vision techniques.

Acknowledgements
The Food-101 data set consists of images from Foodspotting [1] which are not property of the Federal Institute of Technology Zurich (ETHZ). Any use beyond scientific fair use must be negociated with the respective picture owners according to the Foodspotting terms of use [2].
[1] http://www.foodspotting.com/
[2] http://www.foodspotting.com/terms/
====
Target Variable: food_type (object, 101 distinct): ['apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare', 'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito']
====
Features:

food_image (object, 101000 distinct): ['apple_pie/1005649.jpg', 'apple_pie/1011328.jpg', 'apple_pie/101251.jpg', 'apple_pie/1014775.jpg', 'apple_pie/1026328.jpg', 'apple_pie/1028787.jpg', 'apple_pie/1034399.jpg', 'apple_pie/103801.jpg', 'apple_pie/1038694.jpg', 'apple_pie/1043283.jpg']
'''

def load_df(dir_path: str) -> DataFrame:
    df = load_image_food_df(dir_path)
    df = take_10_similar_food_types(df)
    return df


def load_image_food_df(dir_path: str) -> DataFrame:
    ret = []
    images_path = join(dir_path, IMAGE_FOLDER)
    for food_type in sorted(os.listdir(images_path)):
        if not os.path.isdir(join(images_path, food_type)):
            continue
        for img_name in sorted(os.listdir(join(images_path, food_type))):
            img_path = join(food_type, img_name)
            ret.append({IMAGE_FEATURE_NAME: img_path, LABEL_NAME: food_type})
    ret = DataFrame(ret)
    return ret


def take_10_similar_food_types(df: DataFrame) -> DataFrame:
    """
    ['apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare', 'beet_salad', 'beignets', 
    'bibimbap', 'bread_pudding', 'breakfast_burrito', 'bruschetta', 'caesar_salad', 'cannoli', 
    'caprese_salad', 'carrot_cake', 'ceviche', 'cheese_plate', 'cheesecake', 'chicken_curry', 
    'chicken_quesadilla', 'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder', 
    'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes', 'deviled_eggs', 'donuts', 
    'dumplings', 'edamame', 'eggs_benedict', 'escargots', 'falafel', 'filet_mignon', 'fish_and_chips', 
    'foie_gras', 'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice', 
    'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich', 'grilled_salmon', 
    'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros', 'hummus', 
    'ice_cream', 'lasagna', 'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 
    'miso_soup', 'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters', 'pad_thai', 'paella', 'pancakes', 
    'panna_cotta', 'peking_duck', 'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib', 'pulled_pork_sandwich', 
    'ramen', 'ravioli', 'red_velvet_cake', 'risotto', 'samosa', 'sashimi', 'scallops', 'seaweed_salad', 
    'shrimp_and_grits', 'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake', 
    'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare', 'waffles']
    """
    hard_visual_classes = [
    "beef_tartare",
    "tuna_tartare",
    "beef_carpaccio",
    "ceviche",
    "sashimi",
    "scallops",
    "oysters",
    "foie_gras",
    "crab_cakes",
    "lobster_bisque",]
    df = df[df[LABEL_NAME].isin(hard_visual_classes)]
    return df


CONTEXT = "Object recognition of different types of food"
TARGET = CuratedTarget(raw_name=LABEL_NAME, task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name=IMAGE_FEATURE_NAME, feat_type=FeatureType.IMAGE)]
IMAGE_FOLDER = "food-101/food-101/images"
LOADING_FUNC = load_df
