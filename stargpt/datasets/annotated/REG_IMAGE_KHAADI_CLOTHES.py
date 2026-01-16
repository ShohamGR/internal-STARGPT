from ntpath import exists
import os
from os.path import join
from typing import Optional

from pandas import DataFrame

from tabstar2.utils.datasets import load_csv
from tabstar_paper.datasets.curation_objects import CuratedTarget, CuratedFeature
from tabstar_paper.datasets.objects import SupervisedTask, FeatureType



'''
Dataset Name: usman8/khaadis-clothes-data-with-images/
====
Examples: 400
====
URL: https://www.kaggle.com/usman8/khaadis-clothes-data-with-images
====
Description: 
Khaadi's Clothes Data with Images
A comprehensive collection of images and detailed attributes of Khaadi clothing.

About Dataset
This dataset provides a comprehensive collection of high-quality images and detailed attributes of Khaadi's diverse clothing line. It includes a wide range of clothing items such as dresses, shirts, trousers, and more, each accompanied by detailed attributes such as color, size, material, and style. With this dataset, researchers, fashion enthusiasts, and machine learning practitioners can delve into comprehensive analysis, trend identification, and predictive modeling, offering valuable insights into Khaadi's fashion products and customer preferences. Whether for retail analytics, fashion trend analysis, or recommendation systems, this dataset serves as a valuable resource for various analytical and machine learning applications in the domain of fashion and retail.
====
Target Variable: Price (int64, 55 distinct): ['3490', '3990', '2490', '3690', '4990', '4190', '1990', '3190', '2990', '5990']
====
Features:

Product Name (object, 83 distinct): ['Fabrics 3 Piece Suit', 'Fabrics 2 Piece', 'Classic Kameez', 'Classic Kurta', 'Narrow Culottes', 'Dupatta', 'Button Down Shirt', 'Flared Kameez', 'Drop Shoulder', 'Contemporary Kameez']
Product Description (object, 128 distinct): ['Printed | Cambric', 'Dyed Embroidered | Viscose Oak Silk', 'Yarn Dyed Embroidered | Cotton Net', 'Printed | Viscose Crepe', 'Printed | Lawn', 'Printed Lawn | Top Dupatta', 'Dyed Embroidered | Dobby', 'Yarn Dyed | Cotton Polyester Broshia Jacquard', 'Dyed Embroidered | Dull Raw Silk', 'Dyed Embroidered | Viscose Crepe']
Availability (object, 1 distinct): ['In Stock']
Color (object, 29 distinct): ['BLACK', 'BLUE', 'MULTI', 'GREEN', 'OFF-WHITE', 'PINK', 'BEIGE', 'RED', 'WHITE', 'PURPLE']
img_count (int64, 8 distinct): ['5', '6', '2', '8', '7', '4', '3', '1']
img_0 (object, 400 distinct): ['ALK231009/image_0.jpg', 'BLK231004/image_0.jpg', 'ACA231008/image_0.jpg', 'ILK231001/image_0.jpg', 'JK231001/image_0.jpg', 'MLK231001/image_0.jpg', 'AK231006/image_0.jpg', 'BLK231001/image_0.jpg', 'JK231002/image_0.jpg', 'BCH231002/image_0.jpg']
img_1 (object, 399 distinct, 0.2% missing): ['ALK231009/image_1.jpg', 'BLK231004/image_1.jpg', 'ACA231008/image_1.jpg', 'ILK231001/image_1.jpg', 'JK231001/image_1.jpg', 'MLK231001/image_1.jpg', 'AK231006/image_1.jpg', 'BLK231001/image_1.jpg', 'JK231002/image_1.jpg', 'BCH231002/image_1.jpg']
img_2 (object, 344 distinct, 14.0% missing): ['ALK231009/image_2.jpg', 'BLK231004/image_2.jpg', 'ACA231008/image_2.jpg', 'ILK231001/image_2.jpg', 'JK231001/image_2.jpg', 'MLK231001/image_2.jpg', 'AK231006/image_2.jpg', 'JK231002/image_2.jpg', 'BCH231002/image_2.jpg', 'ALK231017/image_2.jpg']
'''

IMG_TEMP_DIR = "Img Path"


def load_df(dir_path: str) -> DataFrame:
    main_dir = join(dir_path, MAIN_DIR)
    df = load_csv(main_dir, "khaadi_data.csv")
    df = process_images(df, dir_path)
    return df


def process_images(df: DataFrame, dir_path: str):
    df[IMG_TEMP_DIR] = df[IMG_TEMP_DIR].apply(lambda img: img.replace("images\\", ""))
    df['img_count'] = df[IMG_TEMP_DIR].apply(lambda i: len(os.listdir(join(dir_path, IMAGE_FOLDER, i))))
    for n in [0, 1, 2]:
        df[f'img_{n}'] = df[IMG_TEMP_DIR].apply(lambda i: join(i, f"image_{n}.jpg"))
        df[f'img_{n}'] = df[f'img_{n}'].apply(lambda i: _remove_if_missing(i, dir_path))
    return df

def _remove_if_missing(img: str, dir_path: str) -> Optional[str]:
    if not exists(join(dir_path, IMAGE_FOLDER, img)):
        return None
    return img


TARGET = CuratedTarget(raw_name='Price', task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = ['ID', 'Product Link', IMG_TEMP_DIR, "Availability"]
TEXT_FEATURES = [CuratedFeature(raw_name='Product Description', feat_type=FeatureType.TEXT)]
FEATURES = [CuratedFeature(raw_name=f"img_{n}", feat_type=FeatureType.IMAGE) for n in [0, 1, 2]] + TEXT_FEATURES
MAIN_DIR = "Khaadi_Data"
IMAGE_FOLDER = join(MAIN_DIR, "images")
LOADING_FUNC = load_df
