import os
from os.path import exists, join

from pandas import DataFrame, read_csv

from tabstar2.utils.datasets import load_csv
from tabstar_paper.datasets.curation_objects import CuratedTarget, CuratedFeature
from tabstar_paper.datasets.objects import SupervisedTask, FeatureType


'''
Dataset Name: musadiqpashak/reddit-memes-dataset/
====
Examples: 4821
====
URL: https://www.kaggle.com/musadiqpashak/reddit-memes-dataset
====
Description: 
The Ultimate Reddit Meme Dataset
Multi-Subreddit Meme Dataset with Metadata and Extracted Text

About Dataset
This dataset contains memes scraped from seven popular meme subreddits on Reddit using the Reddit API. The dataset includes meme images, metadata, and text extracted from the images using OCR (Optical Character Recognition). It is structured to support tasks like meme classification, virality prediction, OCR-based analysis, and temporal meme trend studies. The goal of this dataset is to facilitate analysis on meme trends, virality prediction, and meme categorization based on content and context.

🗂 Subreddits Included

r/Animemes
r/FunnyMemes
r/okbuddyretard
r/ProgrammerHumor
r/dankmemes
r/engineeringmemes
r/shitposting
Source:
Reddit API, scraped using the praw Python package.

Inspiration:
Memes are powerful tools of communication on the internet. By analyzing when, where, and how they are posted and what textual content they carry we can uncover insights into digital culture, humor trends, and audience engagement.

File Information

Files included:

data.csv: Main dataset with metadata and OCR-extracted content.
memes/: Folder containing all meme image files (e.g., JPEG/PNG format).
🧾 Dataset Features

Each meme entry includes:

Title of the Reddit post
Author and their karma details
3.Post timestamp and categorized time of day
4.Number of upvotes, upvote ratio, and comments
5.Image filename and image URL
6.Extracted text from the image using Tesseract OCR
7.Subreddit category and meme context
====
Target Variable: Category (object, 7 distinct): ['ProgrammerHumor', 'Animemes', 'Funnymemes', 'okbuddyretard', 'dankmemes', 'shitposting', 'engineeringmemes']
====
Features:

Title (object, 4503 distinct): ['????????????', '????', '????????', '????LOL', '???', '????????????????', 'Title', 'LOL', 'True story', 'Real']
Author (object, 2436 distinct): ['F3mboyhunter_69', 'Delicious_Maize9656', 'Holofan4life', 'Jackabing', 'PizzaMajestic2634', 'aidantomcy', 'the_forever_wild', 'Ligano_Resurrected', 'YTRKinG', 'MahmoudAO']
Total Karma (int64, 3215 distinct): ['12974036', '124293', '579249', '909765', '23817', '45479', '144587', '66484', '790000', '91282']
Comment Karma (int64, 2456 distinct): ['695', '10820', '366448', '4594', '5042', '19481', '117675', '1735', '30841', '514']
Cake Day (datetime64[ns], 1767 distinct): ['2025-03-07 00:00:00', '2020-08-29 00:00:00', '2016-12-06 00:00:00', '2024-06-28 00:00:00', '2024-09-15 00:00:00', '2023-12-06 00:00:00', '2025-01-14 00:00:00', '2025-03-14 00:00:00', '2025-03-19 00:00:00', '2016-06-16 00:00:00']
Created Time (datetime64[ns], 1508 distinct, 66.8% missing): ['2025-02-04 19:56:00', '2025-01-04 16:21:00', '2025-02-04 01:43:00', '2025-03-04 08:45:00', '2025-03-04 16:45:00', '2025-01-04 19:38:00', '2025-01-04 20:56:00', '2025-03-04 20:14:00', '2025-03-04 08:19:00', '2025-01-04 05:00:00']
Upvotes (int64, 2002 distinct): ['0', '6', '8', '5', '10', '7', '9', '3', '16', '13']
Upvote Ratio (float64, 89 distinct): ['0.99', '0.98', '0.97', '0.96', '0.95', '0.94', '1.0', '0.93', '0.91', '0.92']
Number of Comments (int64, 280 distinct): ['1', '2', '3', '4', '0', '5', '6', '8', '7', '9']
Time of Day (object, 4 distinct): ['Night', 'Afternoon', 'Morning', 'Evening']
File Name (object, 4821 distinct): ['1_SachiMod_Animemes.png', '2_GroovyChirpy_Animemes.jpeg', '3_kf1035_Animemes.jpeg', '4_KaySanTheBrightStar_Animemes.jpeg', '5_Jackabing_Animemes.jpeg', '6_IllustriousFox5135_Animemes.jpeg', '7_iWILLpissINuranus_Animemes.png', '8_Rubikx107_Animemes.png', '9_Prince0x_Animemes.jpeg', '10_MakotoKurume_Animemes.jpeg']
Extracted Text (object, 4778 distinct): ['RIP George Foreman', 'boss battle 1', "When you're accidentally included in a war plan group chat", '3', 'made with mematic', 'ME WAITING FOR MY ERECTION TO CALM DOWN SO | CAN PEE', 'Teachers when a student is getting beaten up in front of them Teachers when that same student starts to fight back', 'MK', 'Engineering students that graduated online in 2020', 'is it really worth it? TruMod']
'''

def load_df(dir_path: str) -> DataFrame:
    main_dir = join(dir_path, MAIN_DIR)
    df = load_csv(main_dir, "data.csv")
    validate_images(df, dir_path)
    return df

def validate_images(df: DataFrame, dir_path: str):
    for filename in df[FILENAME]:
        img_path = join(dir_path, IMAGE_FOLDER, filename)
        if not exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

FILENAME = "File Name"


CONTEXT = ""
TARGET = CuratedTarget(raw_name='Category', task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = ['Post URL']
FEATURES = [CuratedFeature(raw_name=FILENAME, feat_type=FeatureType.IMAGE),
            CuratedFeature(raw_name="Cake Day", feat_type=FeatureType.DATE),
            CuratedFeature(raw_name="Created Time", feat_type=FeatureType.DATE),
            CuratedFeature(raw_name="Extracted Text", feat_type=FeatureType.TEXT),
            CuratedFeature(raw_name="Title", feat_type=FeatureType.TEXT),
            CuratedFeature(raw_name="Author", feat_type=FeatureType.TEXT)]
MAIN_DIR = "reddit_memes_dataset"
IMAGE_FOLDER = join(MAIN_DIR, "memes")
LOADING_FUNC = load_df
