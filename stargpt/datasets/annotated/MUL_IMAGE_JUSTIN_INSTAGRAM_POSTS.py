import os
from os.path import exists, join

from pandas import DataFrame

from tabstar2.utils.datasets import load_csv
from tabstar_paper.datasets.curation_objects import CuratedTarget, CuratedFeature
from tabstar_paper.datasets.objects import SupervisedTask, FeatureType


'''
Dataset Name: aldiandyainf/which-justin-posted-that/
====
Examples: 10319
====
URL: https://www.kaggle.com/aldiandyainf/which-justin-posted-that
====
Description: 
Which Justin posted that?
Practice using tabular, text or image to predict which Justin posted that

About Dataset
Context
The data was gathered using Python by the Instagram GraphQL API (non-documented). All of the images are publicly available.
The reason for this project was to create a fun project where you can practice Computer Vision, Natural Language Processing, and normal tabular machine learning and perhaps combine those models together.

Content
The data contain about ten thousand rows.
For pictures, all of them were thumbnail photos. This included video and carousel posts.

====
Target Variable: username (object, 5 distinct): ['justinbieber', 'justinpjtrudeau', 'justintimberlake', 'justinlong', 'justinhartley']
====
Features:

post_dates (datetime64[ns], 10309 distinct): ['1970-01-01 00:00:01.502211647', '1970-01-01 00:00:01.459043656', '1970-01-01 00:00:01.411856423', '1970-01-01 00:00:01.406165450', '1970-01-01 00:00:01.523773400', '1970-01-01 00:00:01.388026944', '1970-01-01 00:00:01.327745171', '1970-01-01 00:00:01.602533512', '1970-01-01 00:00:01.436279080', '1970-01-01 00:00:01.317930726']
captions (object, 7663 distinct, 19.7% missing): ['📷: @rorykramer', '♛', '@drewhouse', '📸: @rorykramer', '📷: @evanpaterakis', 'GOOD MORNING HERE IS A GUIDED PRAYER TO START YOUR DAY,! FIND A COMFY POSITION AND ENJOY! @churchome @judahsmith', 'Throwback', '#yummy', '📸: @evanpaterakis', 'Lol']
n_likes (int64, 10246 distinct): ['968', '558', '648', '1269784', '35042', '11663', '521', '9408', '31069', '34967']
n_comments (int64, 7858 distinct): ['7', '3', '16', '4', '6', '5', '87', '8', '70', '140']
n_hashtags (int64, 17 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
n_characters (int64, 966 distinct): ['0', '14', '10', '12', '15', '17', '16', '9', '13', '7']
n_words (int64, 292 distinct): ['0', '2', '1', '3', '4', '5', '6', '7', '8', '9']
n_emojis (int64, 15 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '10', '8']
n_mentions (int64, 19 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
is_video (bool, 2 distinct): ['0', '1']
display_picture_url (object, 10319 distinct): ['CbNn8XPrGZG.png', 'CbNnwOfLtow.png', 'CbNmvJErdhZ.png', 'CbNmjmkLVKg.png', 'CbLpmoQPga5.png', 'CbJS1icPNGu.png', 'CbGPVftPpIo.png', 'CbGFsM0Pj1D.png', 'CbDlxuvPHvB.png', 'CbDZ8vhJuiF.png']
'''

IMG_RAW = "display_picture_relative_url"
IMG_NEW = "display_picture_url"

def load_df(dir_path: str) -> DataFrame:
    df = load_csv(dir_path, "dataset.csv")
    df[IMG_NEW] = df[IMG_RAW].apply(lambda img: parse_img(img, dir_path))
    return df

def parse_img(img: str, data_dir: str) -> str:
    img_clean = img.split('/')[-1]
    if not exists(join(data_dir, IMAGE_FOLDER, img_clean)):
        raise FileNotFoundError(f"Image not found: {join(data_dir, IMAGE_FOLDER, img_clean)}")
    return img_clean


CONTEXT = ""
TARGET = CuratedTarget(raw_name='username', task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = [IMG_RAW, 'urls',
                # Too easy with this information
                'n_likes', 'n_comments', 'captions'
                ]
FEATURES = [CuratedFeature(raw_name=IMG_NEW, feat_type=FeatureType.IMAGE),
            CuratedFeature(raw_name="post_dates", feat_type=FeatureType.DATE),
            CuratedFeature(raw_name="captions", feat_type=FeatureType.TEXT),
]
IMAGE_FOLDER = "imgs/imgs"
LOADING_FUNC = load_df
