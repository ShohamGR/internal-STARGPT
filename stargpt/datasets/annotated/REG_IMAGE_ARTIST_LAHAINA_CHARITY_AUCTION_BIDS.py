from os.path import exists, join
from typing import Optional

from pandas import DataFrame

from tabstar2.utils.datasets import load_csv
from tabstar_paper.datasets.curation_objects import CuratedTarget, CuratedFeature
from tabstar_paper.datasets.objects import SupervisedTask, FeatureType


'''
Dataset Name: quillen/artists-for-lahaina-2023/
====
Examples: 968
====
URL: https://www.kaggle.com/quillen/artists-for-lahaina-2023
====
Description: 
Artists for Lahaina 2023
Prices, images, and metadata for all paintings sold at this charity art auction

About Dataset
This dataset covers all paintings sold in the Artists for Lahaina 2023 charity art auction. Plein air painters from around the world donated original artwork to be auctioned online, raising >$500,000 for the Maui Pono Foundation.

About the Auction: The Lahaina Fire on August 8th 2023 was the deadliest in modern American history. The fire burned though 80% of the Hawaiian town and left its surviving residents facing a deeply uncertain future.

Lahaina was once the capital of the Kingdom of Hawaii and it is known for its rich culture & arts community. The town is beloved by the thousands of artists who have found both welcome and inspiration there.

In support of Lahaina’s efforts to rebuild, [Artists for Lahaina organized] an auction of original artwork by artists from all over the world.

All auction proceeds will go to the Maui Pono Foundation, an organization founded by Lahaina families to help their many displaced neighbors recover after the fire. In addition to funding direct relief, the Maui Pono Foundation has generously committed to support efforts to enable artists in West Maui to play a key role in restoring the community they love.

[1]

Note: I am not affiliated with Artists for Lahaina nor the Maui Pono Foundation (although I bid on a few of these paintings).

Header: Keokea View of the Valley - Michael Clements
====
Target Variable: winning_bid (int64, 218 distinct): ['100', '200', '300', '500', '400', '250', '150', '325', '350', '275']
====
Features:

artist (object, 804 distinct): ['Mary Spain', 'Clark Mitchell', 'Morgan Samuel Price', 'Valeh Levy', 'Nic Eason', 'Debra Huse', 'Lisa Mcknett', 'Robert Green', 'Amalia Fisch', 'John Burton']
title (object, 941 distinct, 0.2% missing): ['Lahaina Harbor', 'West Maui Mountains', 'Hope', 'Maui Morning', 'Lahaina Banyan', 'Lahaina Sunset', 'Tucked In', 'Pacific Coast', 'Low Tide', 'Summer Love']
medium (object, 6 distinct): ['Oil', 'Acrylic', 'Watercolor', 'Pastel', 'Other', 'Gouache']
dim1 (float64, 44 distinct, 2.4% missing): ['8.0', '12.0', '11.0', '9.0', '16.0', '6.0', '10.0', '14.0', '20.0', '18.0']
dim2 (float64, 48 distinct, 2.4% missing): ['12.0', '10.0', '14.0', '16.0', '20.0', '8.0', '24.0', '6.0', '9.0', '11.0']
value (float64, 171 distinct, 0.2% missing): ['400.0', '500.0', '300.0', '350.0', '450.0', '1200.0', '800.0', '250.0', '200.0', '900.0']
buy_now (bool, 2 distinct): ['0', '1']
Gallery (object, 3 distinct): ['Gallery Ekolu', 'Gallery Elua', 'Gallery Ekahi']
file_name (object, 965 distinct, 0.3% missing): ['Hamoa Beach, Maui - Aaron Schuerr.jpg', 'Honokahua Bay - Aaron Schuerr.jpg', 'Budding Elegance - Aditi Sharma.jpg', 'West Maui Mountains as viewed from Lahaina - Agata Zbik.jpg', 'Four Shades of Yellow - Aimee Erickson.jpg', 'Happy Hour - Aimee Erickson.jpg', 'After the Storm - Alan Wayne.jpg', 'Breath of Spring - Alan Wolton.jpg', 'Miss Kathy - Alana Knuff.jpg', 'Grazing - Alicia van Thiel.jpg']
'''

FILENAME = "file_name"

def load_df(dir_path: str) -> DataFrame:
    main_dir = join(dir_path, MAIN_DIR)
    df = load_csv(main_dir, "clean_data.csv")
    df[FILENAME] = df[FILENAME].apply(lambda i: validate_images(i, dir_path=dir_path))
    return df


def validate_images(i: str, dir_path: str) -> Optional[str]:
    if not exists(join(dir_path, IMAGE_FOLDER, i)):
        return None
    return i


LABEL_NAME = ""



CONTEXT = ""
TARGET = CuratedTarget(raw_name='winning_bid', task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = ['img_url',]
FEATURES = [CuratedFeature(raw_name=FILENAME, feat_type=FeatureType.IMAGE),
            CuratedFeature(raw_name='artist', feat_type=FeatureType.TEXT),
            CuratedFeature(raw_name='title', feat_type=FeatureType.TEXT),
]
MAIN_DIR = "data"
IMAGE_FOLDER = join(MAIN_DIR, "imgs")
LOADING_FUNC = load_df
