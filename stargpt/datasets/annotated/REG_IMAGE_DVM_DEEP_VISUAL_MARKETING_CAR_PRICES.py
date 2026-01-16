import os
from os.path import exists, join
from typing import Any, Optional

from pandas import DataFrame, read_csv

from tabstar2.utils.datasets import load_csv
from tabstar_paper.datasets.curation_objects import CuratedTarget, CuratedFeature
from tabstar_paper.datasets.objects import SupervisedTask, FeatureType


IMAGE_FEATURE_NAME = "Car Picture"

'''
Dataset Name: osamasaifs/dvm-car-with-images/
====
Examples: 246159
====
URL: https://www.kaggle.com/osamasaifs/dvm-car-with-images
====
Bag Of Tricks Paper:
DVM: Predict the selling price of cars based on car images and the metadata including fuel type,
the number of seats, body type and so on. This dataset originally stems from [36]: https://
deepvisualmarketing.github.io. The original dataset contains multiple metadata tables. We
use Ad table as the metadata, which contains more than 0.25 million used car advertisements. We
further select 25% of the metadata randomly, and split the subset at 3:1 ratio for a new training set
and test set. The original dataset provide car images with different views, and we use the front view.
The license of the original dataset: CC BY-NC 4.0.


Description: 
About Dataset
About Dataset
This dataset is based on the DVM-Car (Deep Visual Marketing – Car) dataset, a large-scale, publicly available dataset designed to support automotive industry research and applications such as car appearance analysis, consumer analytics, sales modeling, and computer vision tasks.

The original DVM-Car dataset combines vehicle images with rich tabular data, enabling both unimodal and multimodal research. This Kaggle release provides the data in its original structure, making it suitable for large-scale experimentation and learning.

Original Dataset Source
The original dataset is published by the Deep Visual Marketing (DVM) project and can be accessed at:

Official Website: https://deepvisualmarketing.github.io/
Please refer to the official website and user manual for detailed documentation of the original data collection process and intended usage.

Data Contents
The dataset consists of both image data and structured tabular data, organized into multiple directories.

1. Vehicle Images
Over 1.45 million vehicle images
Images cover 899 UK market car models
Models span approximately two decades
Images are primarily front-view car images
Includes resized images and quality-checked subsets
Suitable for:
Vehicle classification and recognition
Appearance analysis
Computer vision and deep learning tasks
Image–table multimodal learning
2. Tabular Data
The tabular component contains 62 columns and includes:

Vehicle specifications (model, brand, year, trim)
Sales data spanning multiple years
Pricing information for new vehicles
Advertisement data for used cars
Image metadata such as color, viewpoint, and quality indicators
These tables enable analytical tasks such as sales modeling, consumer behavior analysis, and joint image–tabular research.

3. Directory Structure
confirmed_fronts_clean: quality-checked front-view vehicle images
resized_DVM_clean: resized vehicle images
tables: structured tabular data files
Data Source and Modifications
This dataset originates from the DVM-Car dataset and has been uploaded without preprocessing or data cleaning. No filtering, normalization, resizing, or transformation was applied to either the images or the tabular data.

The only modification performed was renaming files that contained the ' (apostrophe) character, as Kaggle does not support uploading files with this character in filenames. Apart from this filename change, the dataset content and structure remain unchanged.

Potential Use Cases
Vehicle image classification and recognition
Computer vision research
Multimodal learning (image + tabular data)
Automotive market analysis
Academic research and educational projects
Large-scale dataset handling and benchmarking
Citation
If you use this dataset in your work, please cite the original DVM-Car dataset:

DVM-Car: A Large-Scale Dataset for Automotive Applications
Deep Visual Marketing Project
https://deepvisualmarketing.github.io/

Notes
The dataset is large (~16 GB) and may require significant storage and computational resources.
Users are encouraged to consult the official DVM-Car documentation for detailed explanations of the original data schema.
This dataset is provided as-is, with no guarantees regarding completeness or correctness.
License and Usage
The original DVM-Car dataset is publicly available for research and educational purposes.
No explicit open-source license is provided by the original authors.

This Kaggle version is shared for non-commercial research and learning use only.
Users should consult the official DVM-Car website for full usage terms:
https://deepvisualmarketing.github.io/
====
Target Variable: Price (float64, 19741 distinct): ['3995.0', '5995.0', '4995.0', '2995.0', '6995.0', '7995.0', '8995.0', '9995.0', '1995.0', '3495.0']
====
Features:

Maker (object, 84 distinct): ['Ford', 'Audi', 'Vauxhall', 'Volkswagen', 'BMW', 'Nissan', 'Peugeot', 'Toyota', 'Citroen', 'Land Rover']
Genmodel (object, 873 distinct): ['Corsa', 'Focus', 'Fiesta', 'Juke', 'X5', 'Astra', 'Mondeo', 'Golf', '500', 'Kuga']
Adv_year (int64, 10 distinct): ['2018', '2021', '2017', '2020', '2016', '2019', '2015', '2014', '2013', '2012']
Adv_month (int64, 15 distinct): ['5', '8', '4', '7', '6', '3', '2', '1', '12', '11']
Color (object, 22 distinct, 8.2% missing): ['Black', 'Silver', 'Blue', 'Grey', 'White', 'Red', 'Green', 'Yellow', 'Brown', 'Orange']
Reg_year (float64, 25 distinct): ['2015.0', '2017.0', '2016.0', '2014.0', '2018.0', '2013.0', '2012.0', '2011.0', '2019.0', '2010.0']
Bodytype (object, 16 distinct, 0.3% missing): ['Hatchback', 'SUV', 'MPV', 'Saloon', 'Coupe', 'Estate', 'Convertible', 'Pickup', 'Combi Van', 'Panel Van']
Runned_Miles (object, 69651 distinct, 0.3% missing): ['10', '100000', '80000', '60000', '70000', '50000', '90000', '65000', '75000', '4000']
Engin_size (object, 70 distinct, 0.8% missing): ['2.0L', '1.6L', '3.0L', '1.2L', '1.4L', '1.0L', '1.5L', '2.2L', '1.8L', '2.5L']
Gearbox (object, 3 distinct, 0.1% missing): ['Manual', 'Automatic', 'Semi-Automatic']
Fuel_type (object, 13 distinct, 0.1% missing): ['Diesel', 'Petrol', 'Hybrid  Petrol/Electric', 'Electric', 'Hybrid  Petrol/Electric Plug-in', 'Petrol Hybrid', 'Petrol Plug-in Hybrid', 'Hybrid  Diesel/Electric', 'Diesel Hybrid', 'Hybrid  Diesel/Electric Plug-in']
Seat_num (float64, 10 distinct, 2.3% missing): ['5.0', '4.0', '7.0', '2.0', '8.0', '6.0', '9.0', '3.0', '17.0', '1.0']
Door_num (float64, 6 distinct, 1.6% missing): ['5.0', '3.0', '4.0', '2.0', '6.0', '7.0']
Predicted_viewpoint (int64, 9 distinct): ['0', '225', '90', '45', '315', '270', '180', '135', '360']
Quality_check (object, 2 distinct, 74.3% missing): ['P', 'N']
Car Picture (object, 246159 distinct): ['Bentley/Arnage/2000/Silver/Bentley$$Arnage$$2000$$Silver$$10_1$$1$$image_0.jpg', 'Bentley/Arnage/2002/Grey/Bentley$$Arnage$$2002$$Grey$$10_1$$2$$image_0.jpg', 'Bentley/Arnage/2002/Blue/Bentley$$Arnage$$2002$$Blue$$10_1$$3$$image_0.jpg', 'Bentley/Arnage/2003/Green/Bentley$$Arnage$$2003$$Green$$10_1$$4$$image_0.jpg', 'Bentley/Arnage/2003/Grey/Bentley$$Arnage$$2003$$Grey$$10_1$$5$$image_10.jpg', 'Bentley/Arnage/2002/Blue/Bentley$$Arnage$$2002$$Blue$$10_1$$6$$image_0.jpg', 'Bentley/Arnage/2002/Green/Bentley$$Arnage$$2002$$Green$$10_1$$7$$image_0.jpg', 'Bentley/Arnage/2003/Black/Bentley$$Arnage$$2003$$Black$$10_1$$8$$image_0.jpg', 'Bentley/Arnage/2003/Silver/Bentley$$Arnage$$2003$$Silver$$10_1$$9$$image_0.jpg', 'Bentley/Arnage/2003/Green/Bentley$$Arnage$$2003$$Green$$10_1$$11$$image_0.jpg']
'''

def load_df(dir_path: str) -> DataFrame:
    tables_dir = join(dir_path, NUM_DIR, "tables", "tables")
    img_df = get_img_df(tables_dir)
    ad_df = load_csv(tables_dir, "Ad_table.csv")
    df = ad_df.merge(img_df, on='Adv_ID', how='inner')
    df[IMAGE_FEATURE_NAME] = df["Image_name"].apply(lambda x: find_image_name_path(image_name=x, dir_path=dir_path))
    df.columns = [c.strip() for c in df.columns]
    df.drop(columns=['Adv_ID', 'Genmodel_ID', 'Image_ID', 'Image_name'], inplace=True)
    # More/source info
    # basic_df = load_csv(tables_dir, "Basic_table.csv")
    # price_df = load_csv(tables_dir, "Price_table.csv")
    # sales_df = load_csv(tables_dir, "Sales_table.csv")
    # trim_df = load_csv(tables_dir, "Trim_table.csv")
    return df


def get_img_df(tables_dir: str) -> DataFrame:
    img_df = load_csv(tables_dir, "Image_table.csv")
    img_df.columns = [c.strip() for c in img_df.columns]
    img_df["Adv_ID"] = img_df["Image_ID"].str.rsplit("$$", n=1).str[0]
    # There is more than one image per advertisement. For simplicity, let's just take the first one.
    img_df = img_df.groupby("Adv_ID").first().reset_index()
    return img_df


def find_image_name_path(image_name: str, dir_path: str) -> str:
    # Image name looks like: Bentley$$Arnage$$2000$$Silver$$10_1$$1$$image_0.jpg
    brand_model_year_color = "/".join(image_name.split("$$")[:4])
    image_path = join(brand_model_year_color, image_name).replace("'", "")
    full_path = join(dir_path, IMAGE_FOLDER, image_path)
    if not exists(full_path):
        raise FileNotFoundError(f"Image not found: {full_path}")
    return image_path


def normalize_price(price: Any) -> Optional[float]:
    if isinstance(price, (int, float)):
        return price
    if isinstance(price, str) and price.isdigit():
        return int(price)
    if 'ukn' in price.lower() or 'unk' in price.lower():
        return None
    raise ValueError(f"Cannot normalize price: {price}")


CONTEXT = ""
TARGET = CuratedTarget(raw_name="Price", task_type=SupervisedTask.REGRESSION, processing_func=normalize_price)
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name=IMAGE_FEATURE_NAME, feat_type=FeatureType.IMAGE)]
NUM_DIR = "19586296"
IMAGE_FOLDER = "19586296/resized_DVM_clean/resized_DVM_clean"
LOADING_FUNC = load_df
