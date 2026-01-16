import json
import sqlite3
from os.path import join
from typing import Dict, Optional

from pandas import DataFrame, read_sql_query

from tabstar_paper.datasets.curation_objects import CuratedTarget, CuratedFeature
from tabstar_paper.datasets.objects import SupervisedTask, FeatureType

LABEL_NAME = "Total Weight in Pounds"
IMAGE_FEATURE_NAME = "Amazon bin"

'''
Dataset Name: dhruvildave/amazon-bin-image-dataset/
====
Examples: 46405
====
URL: https://www.kaggle.com/dhruvildave/amazon-bin-image-dataset
====
Description:
Amazon Bin Image Dataset
50,000 images and metadata from bins of a pod in an operating Amazon Center

About Dataset
The Amazon Bin Image Dataset contains 50,000 images and metadata from bins of a pod in an operating Amazon Fulfillment Center. The bin images in this dataset are captured as robot units carry pods as part of normal Amazon Fulfillment Center operations. This dataset can be used for research in variety of areas like computer vision, counting genetic items and learning from weakly-tagged data.

For each image, there is a corresponding entry of its metadata in JSON format stored in metadata.sqlite i.e. for image 01290.jpg, there is a corresponding json object in the data field of the metadata file which can be retrieved with query SELECT data FROM metadata WHERE img_id = 01290;

Refer the Starter Notebook to see how to work with the dataset.

Amazon uses a random storage scheme where items are placed into accessible bins with available space, so the contents of each bin are random, rather than organized by specific product types. Thus, each bin image may show only one type of product or a diverse range of products. Occasionally, items are misplaced while being handled, so the contents of some bin images may not match the recorded inventory of that bin.

These are some typical images in the dataset. A bin contains multiple object categories and various number of instances. The corresponding metadata exist for each bin image and it includes the object category identification (ASIN - Amazon Standard Identification Number), quantity and dimensions of objects. The size of bins are various depending on the size of objects in it. The tapes in front of the bins are for preventing the items from falling out of the bins and sometimes it might make the objects unclear. Objects are sometimes heavily occluded by other objects or limited viewpoint of the images.

Image Credits: Unsplash - helloimnik
====
Target Variable: Total Weight in Pounds (float64, 28822 distinct): ['1.0', '1.5', '0.6', '0.5', '1.2', '0.75', '0.9', '0.6', '0.3', '1.2']
====
Features:

Amazon bin (object, 46405 distinct): ['439407.jpg', '337152.jpg', '432703.jpg', '62702.jpg', '531830.jpg', '13249.jpg', '66619.jpg', '241568.jpg', '23919.jpg', '178725.jpg']
product descriptions (object, 45418 distinct): ['TaoTronics TT-AH002 30W Ultrasonic Humidifier with Cool Mist, Classic Dial Knob Control, 3.5L Large Capacity, Two 360 degree Rotatable Outlets', 'Bluedio T2 Plus Turbine Wireless Bluetooth Headphones with Mic/Micro SD Card Slot/FM Radio (Blue)', 'Best LED Bulb Pack of 4 by Vemotix! - 9W equivalent 75W light (3000K) / 600lm - View Angle > 270o- 30.000 Hours Extra Long Lifespan - Very Economic - 100% Satisfaction Guarantee', 'FujiFilm Instax Mini 8 with Strap and Batteries (Blue)', "Ravensburger XXL Children's Globe 180 Piece Puzzleball", "Funny Guy Mugs Shhh There's Wine In Here Ceramic Coffee Mug, White, 11-Ounce", '3M Virtua CCS Protective Eyewear 11872-00000-20, Foam Gasket, Anti Fog Lens, Clear', 'Bayer Advantage II for Large Cats Over 9 lbs, 6 Pack', 'TP-LINK 8-Port Gigabit Ethernet Desktop Switch (TL-SG108)', 'iOttie Easy One Touch 2 Car Mount Holder for iPhone 6s Plus 6s 5s 5c Samsung Galaxy S7 Edge S6 S5 Note 5 4']
Expected Quantity (int64, 79 distinct): ['3', '4', '2', '5', '6', '1', '7', '8', '9', '10']
'''

def get_total_weight(data: Dict) -> Optional[float]:
    items = list(data.values())
    if not items:
        return None
    total_weight = 0.0
    for item in items:
        weight_dict = item.get('weight') or {}
        weight = weight_dict.get('value')
        quantity = item.get('quantity')
        if weight and quantity:
            total_weight += weight * quantity
        else:
            return None
    return total_weight

def parse_product_descriptions(data: Dict) -> str:
    descriptions = []
    for item in data.values():
        name = item.get('name')
        if name:
            descriptions.append(name)
    return "; ".join(descriptions)

def load_df(dir_path: str) -> DataFrame:
    sql_path = join(dir_path, "metadata.sqlite")
    conn = sqlite3.connect(sql_path)
    df = read_sql_query("SELECT * FROM metadata", conn)
    df[IMAGE_FEATURE_NAME] = df['img_id'].apply(lambda i: f"{i}.jpg")
    df.drop(columns=['img_id'], inplace=True)
    df['data'] = df['data'].apply(json.loads)
    key = 'BIN_FCSKU_DATA'
    df[key] = df['data'].apply(lambda d: d.get(key, {}))
    df[LABEL_NAME] = df[key].apply(get_total_weight)
    df['product descriptions'] = df[key].apply(parse_product_descriptions)
    df['Expected Quantity'] = df['data'].apply(lambda d: d.get('EXPECTED_QUANTITY', None))
    df.drop(columns=['data', key], inplace=True)
    return df


CONTEXT = "Amazon Bin Packages from Amazon Center"
TARGET = CuratedTarget(raw_name=LABEL_NAME, task_type=SupervisedTask.REGRESSION)
FEATURES = [CuratedFeature(raw_name=IMAGE_FEATURE_NAME, feat_type=FeatureType.IMAGE),
            CuratedFeature(raw_name='product descriptions', feat_type=FeatureType.TEXT)]
IMAGE_FOLDER = "img"
LOADING_FUNC = load_df
