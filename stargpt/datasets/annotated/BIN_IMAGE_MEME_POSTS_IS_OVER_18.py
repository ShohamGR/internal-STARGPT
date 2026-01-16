from curses import raw
import os
from os.path import exists, join

from pandas import DataFrame, read_csv

from tabstar2.utils.datasets import load_csv
from tabstar_paper.datasets.curation_objects import CuratedTarget, CuratedFeature
from tabstar_paper.datasets.objects import SupervisedTask, FeatureType


'''
Dataset Name: nikitricky/memes/
====
Examples: 5141
====
URL: https://www.kaggle.com/nikitricky/memes
====
Description:
title: r/memes dataset
subtitle: A collection of 7k images scraped from r/memes subreddit
keywords: ['culture and humanities', 'popular culture', 'tabular', 'image']
licenses: [{'name': 'reddit-api'}]
description: This dataset contains a collection of 7053 images that were scraped from the popular r/memes subreddit on Reddit. The images are stored in the images directory, along with a CSV file (post_info.csv) that contains post information such as title, upvotes, comments, and more.

However, it should be noted that approximately 27% of the data could not be scraped, likely due to removal by the subreddit moderators. Nevertheless, this dataset still contains a significant amount of data that can be used for training machine learning models or conducting research in computer vision, natural language processing, or social media analysis.

The images are in various file formats and resolutions and depict a wide range of humorous and satirical content. The post information in the CSV file provides additional context and metadata that can be useful for analysis or classification tasks.

Potential applications of this dataset include training image recognition models to identify different types of memes, analyzing trends and patterns in meme content, or exploring the relationship between meme content and social media engagement.

====
Target Variable: over_18 (int64, 2 distinct): ['0', '1']
====
Features:

title (object, 5058 distinct): ['Pretty much', 'It do be like that', 'Bob | boB', 'Every time', 'True story', '[ Removed by Reddit ]', 'What do you think?', 'It be like that', 'Interesting priorities', 'We live in a society']
score (int64, 1045 distinct): ['0', '20', '18', '6', '7', '14', '12', '17', '1', '3']
upvote_ratio (float64, 81 distinct): ['0.96', '0.97', '0.95', '0.98', '0.94', '0.91', '0.93', '0.92', '0.88', '0.9']
spoiler (int64, 2 distinct): ['0', '1']
created (datetime64[ns], 5131 distinct): ['1970-01-01 00:00:01.682506917', '1970-01-01 00:00:01.682890105', '1970-01-01 00:00:01.682778159', '1970-01-01 00:00:01.682526047', '1970-01-01 00:00:01.682724694', '1970-01-01 00:00:01.682396913', '1970-01-01 00:00:01.682879361', '1970-01-01 00:00:01.682376150', '1970-01-01 00:00:01.682589585', '1970-01-01 00:00:01.683096962']
scrape_time (datetime64[ns], 5123 distinct): ['1970-01-01 00:00:00.000081334', '1970-01-01 00:00:00.001114329', '1970-01-01 00:00:00.000438380', '1970-01-01 00:00:00.000267233', '1970-01-01 00:00:00.000106370', '1970-01-01 00:00:00.000101756', '1970-01-01 00:00:00.000583963', '1970-01-01 00:00:00.000154265', '1970-01-01 00:00:00.000197699', '1970-01-01 00:00:00.000541632']
post_url (object, 5141 distinct): ['12s6x74.png', '12s6yw1.png', '12s6zua.png', '12s71m4.png', '12s72wp.png', '12s774c.png', '12s7jtb.png', '12s7npw.png', '12s7rif.png', '12s7zo0.png']
'''

IMG_COLUMN = "post_url"

def load_df(dir_path: str) -> DataFrame:
    df = load_csv(dir_path, "posts.csv")
    df[IMG_COLUMN] = df['id'].apply(lambda img: get_meme_img(img, dir_path))
    df.columns = [c.strip() for c in df.columns]
    return df

def get_meme_img(img: str, dir_path: str) -> str:
    img_folder = join(dir_path, IMAGE_FOLDER)
    img_filename = f"{img}.png"
    assert exists(join(img_folder, img_filename))
    return img_filename


CONTEXT = ""
TARGET = CuratedTarget(raw_name='over_18', task_type=SupervisedTask.BINARY)
COLS_TO_DROP = ['id', 'edited', 'author_id']
FEATURES = [CuratedFeature(raw_name=IMG_COLUMN, feat_type=FeatureType.IMAGE),
            CuratedFeature(raw_name="scrape_time", feat_type=FeatureType.DATE),
            CuratedFeature(raw_name="created", feat_type=FeatureType.DATE),
            CuratedFeature(raw_name="title", feat_type=FeatureType.TEXT)]
IMAGE_FOLDER = "images"
LOADING_FUNC = load_df
