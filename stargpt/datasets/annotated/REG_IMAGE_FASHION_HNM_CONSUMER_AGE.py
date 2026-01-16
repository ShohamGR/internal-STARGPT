from os.path import exists, join
from typing import Optional

from pandas import DataFrame

from tabstar2.utils.datasets import load_csv
from tabstar_paper.datasets.curation_objects import CuratedTarget, CuratedFeature
from tabstar_paper.datasets.objects import SupervisedTask, FeatureType


'''
Dataset Name: odins0n/handm-dataset-128x128/
====
Examples: 104513
====
URL: https://www.kaggle.com/odins0n/handm-dataset-128x128
====
Description: 

First of all, this is the real competition: https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations
Second, we adjusted a bit the dataset to have a nice predictive task, but this it was not the original one.

Description
H&M Group is a family of brands and businesses with 53 online markets and approximately 4,850 stores. Our online store offers shoppers an extensive selection of products to browse through. But with too many choices, customers might not quickly find what interests them or what they are looking for, and ultimately, they might not make a purchase. To enhance the shopping experience, product recommendations are key. More importantly, helping customers make the right choices also has a positive implications for sustainability, as it reduces returns, and thereby minimizes emissions from transportation.

In this competition, H&M Group invites you to develop product recommendations based on data from previous transactions, as well as from customer and product meta data. The available meta data spans from simple data, such as garment type and customer age, to text data from product descriptions, to image data from garment images.

There are no preconceptions on what information that may be useful – that is for you to find out. If you want to investigate a categorical data type algorithm, or dive into NLP and image processing deep learning, that is up to you.

Dataset Description
For this challenge you are given the purchase history of customers across time, along with supporting metadata. Your challenge is to predict what articles each customer will purchase in the 7-day period immediately after the training data ends. Customer who did not make any purchase during that time are excluded from the scoring.

Files
images/ - a folder of images corresponding to each article_id; images are placed in subfolders starting with the first three digits of the article_id; note, not all article_id values have a corresponding image.
articles.csv - detailed metadata for each article_id available for purchase
customers.csv - metadata for each customer_id in dataset
sample_submission.csv - a sample submission file in the correct format
transactions_train.csv - the training data, consisting of the purchases each customer for each date, as well as additional information. Duplicate rows correspond to multiple purchases of the same item. Your task is to predict the article_ids each customer will purchase during the 7-day period immediately after the training data period.
NOTE: You must make predictions for all customer_id values found in the sample submission. All customers who made purchases during the test period are scored, regardless of whether they had purchase history in the training data.
====
Target Variable: AvgConsumerAge (float64, 55273 distinct): ['36.0', '38.0', '37.0', '34.0', '35.0', '40.0', '39.0', '41.0', '32.0', '33.0']
====
Features:

prod_name (object, 45492 distinct): ['Dragonfly dress', 'Mike tee', 'Wow printed tee 6.99', '1pk Fun', 'TP Paddington Sweater', 'Pria tee', 'Despacito', 'MY', 'Robin 3pk Fancy', 'DANTE set']
product_type_name (object, 130 distinct): ['Trousers', 'Dress', 'Sweater', 'T-shirt', 'Top', 'Blouse', 'Shorts', 'Jacket', 'Shirt', 'Vest top']
product_group_name (object, 19 distinct): ['Garment Upper body', 'Garment Lower body', 'Garment Full body', 'Accessories', 'Underwear', 'Shoes', 'Swimwear', 'Socks & Tights', 'Nightwear', 'Unknown']
graphical_appearance_name (object, 30 distinct): ['Solid', 'All over pattern', 'Melange', 'Stripe', 'Denim', 'Front print', 'Placement print', 'Check', 'Colour blocking', 'Lace']
colour_group_name (object, 50 distinct): ['Black', 'Dark Blue', 'White', 'Light Pink', 'Grey', 'Light Beige', 'Blue', 'Red', 'Light Blue', 'Greenish Khaki']
perceived_colour_value_name (object, 8 distinct): ['Dark', 'Dusty Light', 'Light', 'Medium Dusty', 'Bright', 'Medium', 'Undefined', 'Unknown']
perceived_colour_master_name (object, 20 distinct): ['Black', 'Blue', 'White', 'Pink', 'Grey', 'Red', 'Beige', 'Green', 'Khaki green', 'Yellow']
department_name (object, 250 distinct): ['Jersey', 'Knitwear', 'Trouser', 'Blouse', 'Swimwear', 'Dress', 'Kids Girl Jersey Fancy', 'Expressive Lingerie', 'Young Girl Jersey Fancy', 'Jersey Fancy']
index_name (object, 10 distinct): ['Ladieswear', 'Divided', 'Menswear', 'Children Sizes 92-140', 'Children Sizes 134-170', 'Baby Sizes 50-98', 'Ladies Accessories', 'Lingeries/Tights', 'Children Accessories, Swimwear', 'Sport']
index_group_name (object, 5 distinct): ['Ladieswear', 'Baby/Children', 'Divided', 'Menswear', 'Sport']
section_name (object, 56 distinct): ['Womens Everyday Collection', 'Divided Collection', 'Baby Essentials & Complements', 'Kids Girl', 'Young Girl', 'Womens Lingerie', 'Girls Underwear & Basics', 'Womens Tailoring', 'Kids Boy', 'Womens Small accessories']
garment_group_name (object, 21 distinct): ['Jersey Fancy', 'Accessories', 'Jersey Basic', 'Under-, Nightwear', 'Knitwear', 'Trousers', 'Blouses', 'Shoes', 'Dresses Ladies', 'Outdoor']
detail_desc (object, 43018 distinct, 0.4% missing): ['T-shirt in printed cotton jersey.', 'T-shirt in soft, printed cotton jersey.', 'Leggings in soft organic cotton jersey with an elasticated waist.', 'Fine-knit trainer socks in a soft cotton blend with elasticated tops.', 'Socks in a soft, jacquard-knit cotton blend with elasticated tops.', 'Sunglasses with plastic frames and UV-protective, tinted lenses.', 'Socks in a soft, fine-knit cotton blend with elasticated tops.', 'Boxer shorts in a cotton weave with an elasticated waist, long legs and button fly.', 'Fine-knit socks in a soft cotton blend.', 'Tights in a soft, fine-knit cotton blend with an elasticated waist.']
ProductPic (object, 104072 distinct, 0.4% missing): ['010/0108775015.jpg', '010/0108775044.jpg', '010/0108775051.jpg', '011/0110065001.jpg', '011/0110065002.jpg', '011/0110065011.jpg', '011/0111565001.jpg', '011/0111565003.jpg', '011/0111586001.jpg', '011/0111593001.jpg']
NumberOfSales (float64, 3750 distinct): ['1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0', '10.0']

'''

ARTICLE_ID = "article_id"
PRODUCT_PIC = "ProductPic"
NUMBER_OF_SALES = "NumberOfSales"
AVG_CONSUMER_AGE = "AvgConsumerAge"

def load_df(dir_path: str) -> DataFrame:
    df = load_csv(dir_path, "articles.csv")
    df[PRODUCT_PIC] = df[ARTICLE_ID].apply(lambda r: collect_image(r, dir_path=dir_path))
    df = collect_sales_info(df, dir_path=dir_path)
    return df


def collect_image(article_id: int, dir_path: str) -> Optional[str]:
    # Some pictures don't exist, that is fine
    article_id = _transform_article_id(article_id)
    prefix = article_id[:3]
    path = join(prefix, article_id + ".jpg")
    if not exists(join(dir_path, IMAGE_FOLDER, path)):
        return None
    return path

def collect_sales_info(df: DataFrame, dir_path: str) -> DataFrame:
    df_transactions = load_csv(dir_path, "transactions_train.csv")
    df_transactions[ARTICLE_ID] = df_transactions[ARTICLE_ID].apply(_transform_article_id)
    count_sales = df_transactions.groupby(ARTICLE_ID).size()
    df[ARTICLE_ID] = df[ARTICLE_ID].apply(_transform_article_id)
    df['NumberOfSales'] = df[ARTICLE_ID].map(count_sales).fillna(0)

    df_customers = load_csv(dir_path, "customers.csv")
    df_transactions_with_age = df_transactions[['customer_id', 'article_id']].merge(df_customers[['customer_id', 'age']], on='customer_id')    
    df_article_age = df_transactions_with_age.groupby(ARTICLE_ID).agg({'age': 'mean'}).reset_index()
    df = df.merge(df_article_age, on=ARTICLE_ID, how='inner')
    df.rename(columns={'age': AVG_CONSUMER_AGE}, inplace=True)
    df.drop(columns=[ARTICLE_ID], inplace=True)

    return df


def _transform_article_id(article_id: int) -> str:
    return "0" + str(article_id)


CONTEXT = ""
TARGET = CuratedTarget(raw_name=AVG_CONSUMER_AGE, task_type=SupervisedTask.REGRESSION)
# So many codes and IDs
COLS_TO_DROP = ["product_code", "product_type_no", "graphical_appearance_no", "colour_group_code", 
                "perceived_colour_value_id", "perceived_colour_master_id", "department_no", "index_code", 
                "index_group_no", "section_no", "garment_group_no"]
TEXT_FEATURES = [CuratedFeature(raw_name=f, feat_type=FeatureType.TEXT) for f in ['department_name', 'detail_desc', 'prod_name']]
FEATURES = [CuratedFeature(raw_name=PRODUCT_PIC, feat_type=FeatureType.IMAGE)] + TEXT_FEATURES
IMAGE_FOLDER = "images_128_128"
LOADING_FUNC = load_df
