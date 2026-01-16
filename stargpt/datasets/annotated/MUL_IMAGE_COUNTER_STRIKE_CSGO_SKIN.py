import os
from os.path import join, exists
from typing import Optional

import pandas as pd
from pandas import DataFrame

from tabstar2.utils.datasets import load_csv
from tabstar2.utils.images import download_url_dataset, unzip_url_dataset
from tabstar_paper.datasets.curation_objects import CuratedTarget, CuratedFeature
from tabstar_paper.datasets.objects import SupervisedTask, FeatureType


'''
Dataset Name: MUL_IMAGE_COUNTER_STRIKE_CSGO_SKIN
====
Examples: 956
====
URL: https://figshare.com/ndownloader/files/38077458
====
From MuG Paper: https://aclanthology.org/2023.findings-emnlp.354.pdf

Description: 
Counter-Strike: Global Offensive (CS:GO) is
a multiplayer first-person shooter video game
developed by Valve Corporation and Hidden
Path Entertainment, Inc., where players join
teams to compete in objective-based matches
involving tactical gameplay and precise shooting. CS:GO data is collected from https://www.
csgodatabase.com/.
====
Target Variable: Availability (int64, 4 distinct): ['1', '2', '3', '11']
====
Features:

Skin Name (object, 956 distinct): ['AUG | Spalted Wood', 'SSG 08 | Carbon Fiber', 'PP-Bizon | Chemical Green', 'Sawed-Off | Snake Camo', 'SG 553 | Gator Mesh', 'P90 | Fallout Warning', 'M4A1-S | Moss Quartz', 'FAMAS | CaliCamo', 'M4A1-S | Boreal Forest', 'Sawed-Off | Rust Coat']
Skin Quality (object, 6 distinct): ['Covert', 'Consumer', 'Industrial', 'Mil-spec', 'Restricted', 'Classified']
Skin Category (object, 7 distinct): ['Knife', 'Rifle', 'Pistol', 'SMG', 'Glove', 'Heavy', 'Machine Guns']
Min Price (float64, 773 distinct): ['0.03', '0.09', '0.1', '0.11', '0.05', '0.54', '0.55', '0.06', '0.75', '0.17']
Max Price (float64, 821 distinct, 0.1% missing): ['0.03', '0.12', '0.17', '1.18', '0.08', '0.09', '0.1', '0.21', '0.28', '0.11']
Image Path (object, 956 distinct): ['train_images/30.jpg', 'train_images/415.jpg', 'train_images/354.jpg', 'train_images/385.jpg', 'train_images/405.jpg', 'train_images/336.jpg', 'train_images/179.jpg', 'train_images/93.jpg', 'train_images/171.jpg', 'train_images/383.jpg']
'''

def load_df(dir_path: str) -> DataFrame:
    if not exists(dir_path):
        _download_and_unzip(dir_path=dir_path)
    df = _collect_split_datasets(dir_path=dir_path)
    df['Max Price'] = df['Max Price'].apply(_fix_price)
    df['Min Price'] = df['Min Price'].apply(_fix_price)
    return df

def _download_and_unzip(dir_path: str):
    zip_path = download_url_dataset(url="https://figshare.com/ndownloader/files/38077458", path="CounterStrikeGO")
    unzip_url_dataset(src_path=zip_path, dst_path=dir_path)
    for split in ['train', 'dev', 'test']:
        split_zip_name = join(dir_path, IMAGE_FOLDER, f"{split}_images.zip")
        unzip_url_dataset(src_path=split_zip_name)
        os.remove(split_zip_name)

def _collect_split_datasets(dir_path: str) -> DataFrame:
    main_dir = join(dir_path, IMAGE_FOLDER)
    dfs = []
    for split in ['train', 'dev', 'test']:
        split_df = load_csv(dir_path=main_dir, filename=f"{split}.csv")
        dfs.append(split_df)
    df = pd.concat(dfs)
    return df


def _fix_price(pr: str) -> Optional[float]:
    assert isinstance(pr, str)
    pr = pr.strip()
    pr = pr.replace('$', '').replace(',', '')
    for c in pr:
        if c not in '.0123456789':
            return None
    return float(pr)



CONTEXT = ""
TARGET = CuratedTarget(raw_name="Availability", task_type=SupervisedTask.MULTICLASS, )
COLS_TO_DROP = ["id"]
FEATURES = [CuratedFeature(raw_name="Image Path", feat_type=FeatureType.IMAGE),
            CuratedFeature(raw_name="Skin Name", feat_type=FeatureType.TEXT)]
IMAGE_FOLDER = "CSGO-Skin-quality"
LOADING_FUNC = load_df
