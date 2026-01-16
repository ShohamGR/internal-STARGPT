from os.path import join

from pandas import DataFrame, read_excel

from tabstar2.utils.images import download_url_image_column
from tabstar_paper.datasets.curation_objects import CuratedTarget, CuratedFeature
from tabstar_paper.datasets.objects import SupervisedTask, FeatureType


'''
Dataset Name: simpleaditya/zepto-products-dataset/
====
Examples: 16143
====
URL: https://www.kaggle.com/simpleaditya/zepto-products-dataset
====
Description:
title: Zepto Products Dataset
subtitle: A Comprehensive Dataset of Products, Prices, and Promotions on the Zepto App
keywords: ['business', 'tabular', 'image', 'text', 'english']
licenses: [{'name': 'unknown'}]
description: **Zepto Product Dataset**
A comprehensive dataset of products available on Zepto, a popular grocery delivery service.

**Description**
This dataset contains information about products available on the grocery delivery platform, Zepto. The data was collected using Selenium, a web scraping tool. It is organized into two CSV files:

1. Zepto.csv: This file contains a comprehensive list of products available on the standard Zepto interface.
2. Zepto Super saver.csv: This file contains products that are part of the "Super Saver" deals, often featuring discounts and special offers.

This dataset can be used for various analytical purposes, such as:

**Price analysis**: Compare product prices, analyze discount strategies, and identify pricing trends.
**Customer sentiment analysis**: Analyze customer ratings and reviews to understand product popularity and customer satisfaction.
**Sales prediction**: Build models to predict product demand and availability.
**Product categorization**: Explore the hierarchical structure of product categories and sub-categories.

**Files**
1. Zepto.csv: A dataset of products from the standard Zepto interface.
2. Zepto Super saver.csv: A dataset of products with special discounts and offers from the Zepto Super Saver interface.

**Columns**
Both files have the following columns:

1. Image: The URL of the product image.
2. Name: The name of the product.
3. Price: The selling price of the product.
4. Original Price: The original price of the product before any discounts.
5. Ratings: The customer rating of the product.
6. Quantity: The quantity or volume of the product (e.g., 1kg, 500ml).
7. Status: The availability status of the product (e.g., "Available", "Sold Out").
8. Review: The number of reviews for the product.
9. Sub-Category: The sub-category to which the product belongs.
10. Category: The main category to which the product belongs.
11. Interface: The Zepto interface from which the data was scraped.

**Acknowledgements**
This data was scraped from the Zepto website. Please acknowledge Zepto as the source of the data in any publications or analyses.
====
Target Variable: Status (object, 2 distinct): ['Available', 'Sold Out']
====
Features:

Image (object, 13345 distinct): ['https___cdn.zeptonow.com_production_image_not_available.jpeg', 'https___cdn.zeptonow.com_production_tr_w-1280,ar-1000-1000,pr-true,f-auto,q-80_cms_product_variant_1f54fb59-8787-4941-acc0-bd51e2e5c6f3.jpeg', 'https___cdn.zeptonow.com_production_tr_w-1280,ar-5198-5198,pr-true,f-auto,q-80_cms_product_variant_0e21d490-bc56-404a-8686-31329201fc13.jpeg', 'https___cdn.zeptonow.com_production_tr_w-1280,ar-2400-2400,pr-true,f-auto,q-80_cms_product_variant_f8da00eb-40a6-4244-8b96-cb3f0b7cb809.jpeg', 'https___cdn.zeptonow.com_production_tr_w-1280,ar-2000-2000,pr-true,f-auto,q-80_cms_product_variant_998928b5-a6f5-4d37-9e32-71c981261bc2.jpeg', 'https___cdn.zeptonow.com_production_tr_w-1280,ar-1024-1024,pr-true,f-auto,q-80_cms_product_variant_115cdb72-3cc4-4daf-937b-f4b0e91ad35e.jpeg', 'https___cdn.zeptonow.com_production_tr_w-1280,ar-1200-1200,pr-true,f-auto,q-80_cms_product_variant_4a98bb78-1bf5-4967-9cae-60fc5fe82512.jpeg', 'https___cdn.zeptonow.com_production_tr_w-1280,ar-2100-2100,pr-true,f-auto,q-80_cms_product_variant_e9b99690-56ce-45c5-9552-f843f0950b4e.jpeg', 'https___cdn.zeptonow.com_production_tr_w-1280,ar-1024-1024,pr-true,f-auto,q-80_cms_product_variant_767a1090-3f0f-414f-a134-9b7da9d0e757.jpeg', 'https___cdn.zeptonow.com_production_tr_w-1280,ar-1200-1200,pr-true,f-auto,q-80_cms_product_variant_b06e8407-0343-4e47-a029-1747abd07a41.jpg']
Name (object, 11999 distinct): ['Kinley Soda Pet 750 ml Combo', 'Bisleri Packaged Drinking Water', 'Nescafe Classic - Instant Coffee Powder - 100% Pure Coffee', 'Omnigel Gel', "Lay's India's Magic Masala Potato Chips", 'Gatorade Orange Zero Sugar 500 ml Combo', 'Kinley Packaged Drinking Water 1 l Combo', 'Bauli Moonfils Croissants Choco Cream (Egg) 50 g Combo', "Lay's American Cream & Onion Potato Chips", 'Nescafe Gold Blend Rich and Smooth Arabica and Robusta Instant Coffee']
Price (int64, 1162 distinct): ['99', '90', '59', '50', '120', '100', '80', '49', '129', '54']
Original Price (float64, 970 distinct, 11.0% missing): ['60.0', '120.0', '100.0', '150.0', '99.0', '299.0', '50.0', '199.0', '90.0', '499.0']
Ratings (float64, 13 distinct): ['4.7', '4.6', '4.5', '4.4', '4.3', '4.2', '4.8', '4.9', '5.0', '4.1']
Review (int64, 1123 distinct): ['59', '56', '58', '53', '55', '54', '52', '61', '57', '69']
Quantity (object, 1457 distinct): ['1 pc', '100 g', '500 g', '2 combo', '200 g', '1 kg', '1 L', '250 ml', '50 g', '250 g']
Sub-Category (object, 201 distinct): ['Top deals', 'Tea & Coffee', 'Zepto Cafe', 'Powders & Pastes', 'Vitamin Supplements', 'Mints & Gums', 'Chips & Crisps', 'Skincare', 'Pooja & Worship Needs', 'Oil']
Category (object, 22 distinct): ['Cold Drinks & Juices', 'Home Needs', 'Atta, Rice, Oil & Dals', 'Health & Baby Care', 'Breakfast & Sauces', 'Bath & Body', 'Masala & Dry Fruits', 'Munchies', 'Makeup & Beauty', 'Sweet Cravings']
'''

def load_df(dir_path: str) -> DataFrame:
    df = read_excel(join(dir_path, "zepto dataset.xlsx"))
    img_folder = join(dir_path, IMAGE_FOLDER)
    df = download_url_image_column(df=df, img_folder=img_folder, img_col=IMAGE_FEATURE_NAME)
    return df

IMAGE_FEATURE_NAME = "Image"


CONTEXT = ""
TARGET = CuratedTarget(raw_name='Status', task_type=SupervisedTask.BINARY)
COLS_TO_DROP = ["Interface"]
TEXT_FETURES = [CuratedFeature(raw_name=f, feat_type=FeatureType.TEXT) for f in ['Name', 'Quantity', 'Sub-Category']]
FEATURES = [CuratedFeature(raw_name=IMAGE_FEATURE_NAME, feat_type=FeatureType.IMAGE)]
IMAGE_FOLDER = "downloaded_zepto_images"
LOADING_FUNC = load_df
