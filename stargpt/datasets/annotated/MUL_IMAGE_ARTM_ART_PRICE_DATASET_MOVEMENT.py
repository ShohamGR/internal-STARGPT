import os
from os.path import exists, join
from typing import Counter

from pandas import DataFrame

from tabstar2.utils.datasets import load_csv
from tabstar_paper.datasets.curation_objects import CuratedTarget, CuratedFeature
from tabstar_paper.datasets.objects import SupervisedTask, FeatureType

MOVEMENT = "movement"
IMAGE_FEATURE_NAME = "Art Picture"

'''
Dataset Name: flkuhm/art-price-dataset/
====
Examples: 754
====
URL: https://www.kaggle.com/flkuhm/art-price-dataset
====
Bag Of Tricks Paper:
artm: Predict the movement of an artwork (34 total) based on its image and metadata including
title, price, condition and so on. This dataset originally stems from https://www.kaggle.com/
datasets/flkuhm/art-price-dataset. We randomly split the data at 3:1 ratio for new training
set and test set. The license of the original dataset: CC BY-NC-SA 4.0.

Description: 
About Dataset
In addition to images of artworks and sculptures that have been offered for sale on Sothebys, the dataset also contains further information about the associated artworks, such as their price or associated movement.
====
Target Variable: movement (object, 11 distinct): ['Realism', 'Abstract', 'Expressionism', 'Pop Art', 'Conceptual ', 'Other', 'Surrealism', 'Impressionism', 'Geometric Abstraction', 'Minimalism']
====
Features:

price (float64, 108 distinct): ['800.0', '1.5', '680.0', '3.0', '1.275', '5.0', '6.0', '15.0', '4.0', '2.5']
artist (object, 454 distinct, 0.1% missing): ['Russell Young', 'John Fischer', 'Ruth Bernhard', 'Donald Sultan', 'Richard Bernstein', 'Ed  Ruscha', 'Grant Hacking', 'Kim Gottlieb Walker', 'Robert Indiana', 'Cindy  Sherman']
title (object, 679 distinct): ['Untitled', 'Untitled ', 'Madame de Pompadour (née Poisson)', 'Rays', 'Fragments Of Hope', 'Sitelines', 'Refinery Road (one of three)', 'Elizabeth Taylor Portrait', 'Audrey Hepburn', 'Seasons of Hope, Autumn']
yearCreation (object, 136 distinct): ['2012', '1990', '1989', '2008', '1986', '2021', '[nan]', '2016', '2014', '2011']
signed (object, 390 distinct): ['[nan]', 'Signed lower right', 'Signed verso', 'Signed lower right recto', 'Signed and dated lower right recto', 'Signed by artist and numbered, recto', 'Signed and titled bottom right corner ', 'Signed and numbered recto', 'Signed and numbered', 'Signed and numbered verso']
condition (object, 376 distinct): ['Excellent condition.', 'The work is in excellent condition, direct from the publisher.', 'This work is in excellent condition, direct from the publisher.', 'This work is in very good condition.Not examined out of frame.No obvious signs of wear to art.', 'This work is in excellent condition.', 'Very good condition', 'This work is in very good condition.Not examined outside of frame.No obvious signs of damage to artwork.', 'The work is in excellent condition.', 'This work is in very good condition.No visible signs of wear to artwork.Not examined out of frame.', 'This work is in very good condition.Artwork not examined outside of frame.No obvious signs of wear to artwork.']
period (object, 5 distinct): ['Contemporary', 'Post-War', 'Modern', '19th Century', '[nan]']
Art Picture (object, 754 distinct): ['image_1.png', 'image_2.png', 'image_3.png', 'image_4.png', 'image_5.png', 'image_6.png', 'image_7.png', 'image_8.png', 'image_9.png', 'image_10.png']
'''

def load_df(dir_path: str) -> DataFrame:
    df = load_csv(dir_path, "artDataset.csv")
    add_images(df, dir_path=dir_path)
    normalize_infrequent_movements(df)
    df.drop(columns=['Unnamed: 0'], inplace=True)
    return df


def add_images(df: DataFrame, dir_path: str):
    df[IMAGE_FEATURE_NAME] = [f"image_{n+1}.png" for n in range(len(df))]
    for img in df[IMAGE_FEATURE_NAME]:
        img_path = join(dir_path, IMAGE_FOLDER, img)
        if not exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

def fix_price_usd(p: str) -> float:
    # Examples: ['800 USD', '1.500 USD', '680 USD', '3.000 USD', '1.275 USD', '5.000 USD', '6.000 USD', '15.000 USD', '4.000 USD', '2.500 USD']
    if p.endswith(' USD'):
        p = p[:-4]
    return float(p)



def normalize_infrequent_movements(df: DataFrame):
    """
    movement
    Realism                                  177
    Abstract                                 153
    Expressionism                            103
    Pop Art                                   88
    Conceptual                                73
    Surrealism                                21
    Impressionism                             20
    Geometric Abstraction                     19
    Minimalism                                18
    Abstract Expressionism                    16
    Feminist Art                               7
    Traditional                                5
    Organic/Biomorphic Abstraction             5
    Nouveau Réalisme                           4
    [nan]                                      4
    Post-Minimalism                            4
    Post-Impressionism                         4
    Social Realism                             4
    Photorealism                               4
    Modernism                                  3
    Performance Art                            3
    Street Art                                 3
    Environmental Art                          3
    Romanticism                                2
    Punk                                       2
    Baroque                                    1
    Punk, Young British Artists, Abstract      1
    Neo-Expressionism                          1
    Magic Realism                              1
    Neogeo                                     1
    Art Deco                                   1
    Cubism                                     1
    Art Brut                                   1
    Art Nouveau                                1
    """
    movement_cnt = Counter(df[MOVEMENT])
    infrequent_movements = [m for m, cnt in movement_cnt.items() if cnt < 18]
    df[MOVEMENT] = df[MOVEMENT].apply(lambda x: "Other" if x in infrequent_movements else x)



def fix_price_usd(p: str) -> float:
    # Examples: ['800 USD', '1.500 USD', '680 USD', '3.000 USD', '1.275 USD', '5.000 USD', '6.000 USD', '15.000 USD', '4.000 USD', '2.500 USD']
    if p.endswith(' USD'):
        p = p[:-4]
    return float(p)


CONTEXT = ""
TARGET = CuratedTarget(raw_name=MOVEMENT, task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = []
TEXT_FEATURES = ['artist', 'condition', 'signed', 'title', 'yearCreation']
FEATURES = [CuratedFeature(raw_name=IMAGE_FEATURE_NAME, feat_type=FeatureType.IMAGE),
            CuratedFeature(raw_name="price", processing_func=fix_price_usd)] + [CuratedFeature(raw_name=tf, feat_type=FeatureType.TEXT) for tf in TEXT_FEATURES]
IMAGE_FOLDER = "artDataset"
LOADING_FUNC = load_df