from os.path import join

from pandas import DataFrame

from tabstar2.utils.datasets import load_csv
from tabstar2.utils.images import download_url_image_column
from tabstar_paper.datasets.curation_objects import CuratedTarget, CuratedFeature
from tabstar_paper.datasets.objects import SupervisedTask, FeatureType


'''
Dataset Name: furduisorinoctavian/romanian-products-from-emag-and-flanco/
====
Examples: 50000
====
URL: https://www.kaggle.com/furduisorinoctavian/romanian-products-from-emag-and-flanco
====
Description:
title: Romanian products from Emag and Flanco
subtitle: Names, descriptions, prices, ratings and URLs for more than 80 000 products
keywords: ['electronics', 'tabular', 'image', 'text', 'romanian']
licenses: [{'name': 'MIT'}]
description: ##**Important**
Every product was crawled from Emag and Flanco websites in November 2023. Prices, ratings and URLs can differ or may not exist anymore. 😶

##**Dataset columns and meanings**
This dataset contains 85766 rows and 8 columns:
1. id - Unique identifier auto-increment ( integer )
2. name - the name of the product as it is on the web page ( string )
3. price - the price of the product ( float numeric value )
4. rating - the rating of the product ( from 0 to 5 )
5. url - the URL where you can find the product ( string )
6. image_url - the URL where you can find the image of the product ( string )
7. shop - the shop where you can find the product ( string ) - only contains EMAG and FLANCO, can be easily changed to integers like 1 = EMAG and 2 = FLANCO
8. category - the category of the product ( string ) - there are 83 different categories

Some of the most important categories are:
- Laptops
- Phones
- PCs and peripherals
- Appliances
- Gaming stuff
- TVs
- Cameras

## Research ideas:

- Price Analysis Across Categories
- Rating Distribution
- Shop Comparison
- Category Popularity
- Product Diversity
- Correlation Between Price and Rating
- Visual Analysis
- Product Recommendations
- Competitive Pricing Analysis
- Image recognition for products like laptops, TVs, smartphones etc.
====
Target Variable: rating (float64, 247 distinct): ['0.0', '5.0', '4.0', '3.0', '4.5', '1.0', '4.67', '4.33', '4.75', '2.0']
====
Features:

name (object, 47534 distinct): ['Rucsac laptop, Poliester, 15.6 inch, Negru', 'Husa casti wireless, Silicon, Compatibila cu Airpods 1/2, Multicolor', 'Suport pentru laptop Misura, Reglabil, Argintiu', 'Rucsac laptop, Poliester, 15.6 inch, Gri', 'Rucsac laptop, Poliester, 15.6 inch, Albastru', 'Geanta pentru laptop, Poliester, Durabil, Unisex, Gri', 'Husa casti wireless, Silicon, Compatibila cu Airpods Pro, Multicolor', 'Adaptor incarcator CA, Lenovo, Negru', 'Husa laptop, Poliester, 13.3 inch, Gri', 'Hard Disk, Hewlett Packard, 600 GB, SAS, Multicolor']
price (float64, 17843 distinct, 0.8% missing): ['22600.0', '12870.0', '28199.0', '11900.0', '13640.0', '5900.0', '14100.0', '23800.0', '19800.0', '12100.0']
image_url (object, 38760 distinct): ['https___s13emagst.akamaized.net_products_1885_1884484_images_res_62ca8bf9bef23b438843107662fc5f34.jpg?width=720&height=720&hash=1CE1BBC1CBF2A959DB8EF9FFB149AFC4', 'https___s13emagst.akamaized.net_products_1885_1884284_images_res_afba0c4b6bf11bf309487d7142fd4841.jpg?width=720&height=720&hash=92201387CD7496BC046A9B2C58365F6B', 'https___s13emagst.akamaized.net_products_2007_2006050_images_res_58eefaa5bb3c9cd17c174527a84fd91c.jpg?width=720&height=720&hash=E2699AFC1DC86FF68B18E9575F08EDC5', 'https___s13emagst.akamaized.net_products_23464_23463898_images_res_502968c5db7b63a56c6611d2c208df7d.jpg?width=720&height=720&hash=2BEC38D493A363704E53C9CFA60C0B85', 'https___s13emagst.akamaized.net_products_2006_2005193_images_res_a06da35d2d227b01318193de52688721.jpg?width=720&height=720&hash=BC7D269DACC375E3ECEB85D4DD937DC5', 'https___s13emagst.akamaized.net_products_2007_2006202_images_res_1e4a0ab813d1dc7b7a413d80f0762ae9.jpg?width=720&height=720&hash=7388AEA96972EEA1E6F7D605D5DAB946', 'https___s13emagst.akamaized.net_products_8724_8723105_images_res_bd9d4bfe2546255f88b1de98a277cdce.jpg?width=720&height=720&hash=B890317156BFCCC40AF3D30899390F9C', 'https___s13emagst.akamaized.net_products_3929_3928802_images_res_82ebf3d8f7eb64e274c5c34dddd24143.jpg?width=720&height=720&hash=678F5A483B51CCF4B8D567A38E7D8E58', 'https___s13emagst.akamaized.net_products_1886_1885409_images_res_12062f6997de99b1f39e86fdfa6f75eb.jpg?width=720&height=720&hash=1A2268FEA5634409B7F35ED853E174BD', 'https___s13emagst.akamaized.net_products_6127_6126882_images_res_d4bbd356a401a793b63d85b6728c243b.jpg?width=720&height=720&hash=BEDB568E10E13FE94366C14726358033']
category (object, 35 distinct): ['genti laptop', 'laptop', 'baterii laptop', 'incarcatoare laptop', 'accesorii laptop', 'standuri coolere', 'baterii-telefoane', 'placi_baza', 'alte-accesorii-telefoane', 'carduri-memorie']
'''

URL = "image_url"

def load_df(dir_path: str) -> DataFrame:
    df = load_csv(dir_path, "products.csv")
    # Take the top 50K rows, whatever. It's too much to run everything at once.
    df = df[:50000]
    img_folder = join(dir_path, IMAGE_FOLDER)
    df = download_url_image_column(df=df, img_folder=img_folder, img_col=URL)
    return df

def parse_price(price: str) -> float | None:
    # '226,00 ', '128,70 ', '281,99 '
    assert isinstance(price, str)
    # If there are non numbers:
    try:
        price = price.replace(',', '').replace(' ', '').replace('.', '').strip()
        return float(price)
    except ValueError:
        return None

CONTEXT = ""
TARGET = CuratedTarget(raw_name='rating', task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = ['url', 'shop', 'id']
FEATURES = [CuratedFeature(raw_name=URL, feat_type=FeatureType.IMAGE),
            CuratedFeature(raw_name='price', processing_func=parse_price, feat_type=FeatureType.NUMERIC),
            CuratedFeature(raw_name='name', feat_type=FeatureType.TEXT),
]
IMAGE_FOLDER = "downloaded_romania_images"
LOADING_FUNC = load_df
