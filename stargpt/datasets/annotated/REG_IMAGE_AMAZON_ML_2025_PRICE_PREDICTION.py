from os.path import join

import pandas as pd
from pandas import DataFrame

from tabstar2.utils.images import download_url_image_column
from tabstar_paper.datasets.curation_objects import CuratedTarget, CuratedFeature
from tabstar_paper.datasets.objects import SupervisedTask, FeatureType


'''
Dataset Name: raghavdharwal/amazon-ml-challenge-2025/
====
Examples: 74999
====
URL: https://www.kaggle.com/raghavdharwal/amazon-ml-challenge-2025
====
Description:
About Dataset
ML Challenge 2025 Problem Statement
Smart Product Pricing Challenge
In e-commerce, determining the optimal price point for products is crucial for marketplace success and customer satisfaction. Your challenge is to develop an ML solution that analyzes product details and predict the price of the product. The relationship between product attributes and pricing is complex - with factors like brand, specifications, product quantity directly influence pricing. Your task is to build a model that can analyze these product details holistically and suggest an optimal price.

Data Description:
The dataset consists of the following columns:

sample_id: A unique identifier for the input sample
catalog_content: Text field containing title, product description and an Item Pack Quantity(IPQ) concatenated.
image_link: Public URL where the product image is available for download.
Example link - https://m.media-amazon.com/images/I/71XfHPR36-L.jpg
To download images use download_images function from src/utils.py. See sample code in src/test.ipynb.
price: Price of the product (Target variable - only available in training data)
Dataset Details:
Training Dataset: 75k products with complete product details and prices
Test Set: 75k products for final evaluation
Output Format:
The output file should be a CSV with 2 columns:

sample_id: The unique identifier of the data sample. Note the ID should match the test record sample_id.
price: A float value representing the predicted price of the product.
Note: Make sure to output a prediction for all sample IDs. If you have less/more number of output samples in the output file as compared to test.csv, your output won't be evaluated.

File Descriptions:
Source files

src/utils.py: Contains helper functions for downloading images from the image_link. You may need to retry a few times to download all images due to possible throttling issues.
sample_code.py: Sample dummy code that can generate an output file in the given format. Usage of this file is optional.
Dataset files

dataset/train.csv: Training file with labels (price).
dataset/test.csv: Test file without output labels (price). Generate predictions using your model/solution on this file's data and format the output file to match sample_test_out.csv
dataset/sample_test.csv: Sample test input file.
dataset/sample_test_out.csv: Sample outputs for sample_test.csv. The output for test.csv must be formatted in the exact same way. Note: The predictions in the file might not be correct
Constraints:
You will be provided with a sample output file. Format your output to match the sample output file exactly.

Predicted prices must be positive float values.

Final model should be a MIT/Apache 2.0 License model and up to 8 Billion parameters.

Evaluation Criteria:
Submissions are evaluated using Symmetric Mean Absolute Percentage Error (SMAPE): A statistical measure that expresses the relative difference between predicted and actual values as a percentage, while treating positive and negative errors equally.

Formula:

SMAPE = (1/n) * Σ |predicted_price - actual_price| / ((|actual_price| + |predicted_price|)/2)
Example: If actual price = $100 and predicted price = $120
SMAPE = |100-120| / ((|100| + |120|)/2) * 100% = 18.18%

Note: SMAPE is bounded between 0% and 200%. Lower values indicate better performance.

Leaderboard Information:
Public Leaderboard: During the challenge, rankings will be based on 25K samples from the test set to provide real-time feedback on your model's performance.
Final Rankings: The final decision will be based on performance on the complete 75K test set along with provided documentation of the proposed approach by the teams.
Submission Requirements:
Upload a test_out.csv file in the Portal with the exact same formatting as sample_test_out.csv

All participating teams must also provide a 1-page document describing:

Methodology used
Model architecture/algorithms selected
Feature engineering techniques applied
Any other relevant information about the approach
Note: A sample template for this documentation is provided in Documentation_template.md
Academic Integrity and Fair Play:
⚠️ STRICTLY PROHIBITED: External Price Lookup

Participants are STRICTLY NOT ALLOWED to obtain prices from the internet, external databases, or any sources outside the provided dataset. This includes but is not limited to:

Web scraping product prices from e-commerce websites
Using APIs to fetch current market prices
Manual price lookup from online sources
Using any external pricing databases or services
Enforcement:

All submitted approaches, methodologies, and code pipelines will be thoroughly reviewed and verified
Any evidence of external price lookup or data augmentation from internet sources will result in immediate disqualification
Fair Play: This challenge is designed to test your machine learning and data science skills using only the provided training data. External price lookup defeats the purpose of the challenge.

Tips for Success:
Consider both textual features (catalog_content) and visual features (product images)
Explore feature engineering techniques for text and image data
Consider ensemble methods combining different model types
Pay attention to outliers and data preprocessing
====
Target Variable: price (float64, 11862 distinct): ['14.99', '9.99', '19.99', '12.99', '8.99', '7.99', '13.99', '11.99', '16.99', '17.99']
====
Features:

catalog_content (object, 74899 distinct): ['Item Name: PAPYRUS Everyday Card, 1 EA\nValue: 1.0\nUnit: Count\n', "Item Name: French's White Cheddar Crispy Fried Onions, 6 oz\nBullet Point 1: Made with real onions\nBullet Point 2: White cheddar and onion-flavored topping with a crispy, crunchy texture\nBullet Point 3: Resealable package for freshness\nBullet Point 4: Kosher certified; product of the USA\nBullet Point 5: Add a pop of cheesy, fried onion taste and texture to salads, soups and loaded mashed potatoes\nValue: 6.0\nUnit: Ounce\n", "Item Name: Rose’s Sweetened Grenadine Syrup 25oz Bottle | Perfect for Cocktails, Beverages, and Mixers\nBullet Point 1: Grenadine: A rich, pomegranate flavor, sweetened for perfect mixability in your favorite cocktails and spirits. It's also an ideal addition to non-alcoholic drinks, adding both a vibrant splash of color and delicious flavor.\nBullet Point 2: Trusted for Over a Century: Rose’s has been the go-to brand for premium mixers for more than 100 years, earning the trust of bartenders and cocktail enthusiasts alike.\nBullet Point 3: Exceptional Quality: Renowned for its consistent, high-quality mixers, Rose’s ensures your cocktails are always perfectly balanced and flavorful.\nBullet Point 4: Versatile Flavor Range: Available in a variety of flavors, including Lime, Grenadine, Peach, Blueberry, Sweet and Sour, Strawberry, and Simple Syrup, to complement a wide array of cocktail recipes.\nBullet Point 5: Perfect for Classic & Modern Cocktails: Whether you're mixing timeless drinks or creating new concoctions, Rose’s mixers add a touch of excellence to every beverage.\nValue: 25.0\nUnit: Fl Oz\n", 'Item Name: PAPYRUS Everyday Card, 1 EA\nProduct Description: Lifestyle\nValue: 1.0\nUnit: Count\n', 'Item Name: Eden Foods Salt Gomasio Ssme Seawd\nValue: 1.0\nUnit: Count\n', 'Item Name: Arrowhead Mills Organic Yellow Popcorn, 28 oz Bag\nBullet Point 1: 1 - 28 oz Bag of Arrowhead Mills Organic Yellow Popcorn\nBullet Point 2: Arrowhead Mills Organic Yellow Popcorn is a classic snack favorite for all ages\nBullet Point 3: Made of organic yellow popcorn kernels; Great for a delicious whole grain snack\nBullet Point 4: Our organic yellow variety can be popped and made into popcorn balls\nBullet Point 5: Great for making "Popcorn Supreme” – a crunchy combination of seeds, nuts, honey, and of course, popcorn\nValue: 28.0\nUnit: Ounce\n', 'Item Name: McCormick Golden Dipt Cracker Meal Seafood Fry Mix, 10 oz (Pack of 8)\nBullet Point 1: Wheat flour-based fry mix gives seafood a crumb coating that seals in juices\nBullet Point 2: 3 easy steps: lightly moisten fish, coat with Cracker Meal, then fry\nBullet Point 3: Tasty with tilapia, shrimp, scallops, shucked oysters and soft-shell crabs\nBullet Point 4: Substitute Cracker Meal for bread crumbs in meat loaf or meat balls\nBullet Point 5: Dip fish in a mixture of 3 tbsp milk and 2 beaten eggs for a thicker coating\nValue: 80.0\nUnit: Ounce\n', 'Item Name: ABSOLUTELY GLUTEN FREE FLATBREAD GF TSTD ONION, 5.29 OZ, PK- 12\nValue: 1.0\nUnit: Count\n', 'Item Name: BARNEY BUTTER NUT BTTR ALMND CRNCHY, 16 OZ\nValue: 16.0\nUnit: Fl Oz\n', 'Item Name: McCormick Golden Dipt Fish Fry Seafood Fry Mix, 10 oz (Pack of 8)\nBullet Point 1: Seasoned breading made with flour and corn meal for crisp fried fish\nBullet Point 2: Features McCormick spices like celery seed, black pepper and garlic\nBullet Point 3: 3 easy steps: moisten fish in water, coat in Fry Mix, fry in minutes to golden brown\nBullet Point 4: Use as fish sticks breading for kid-friendly dinner\nBullet Point 5: Directions for deep-frying or oven-frying\nValue: 80.0\nUnit: Ounce\n']
product_image (object, 72287 distinct): ['https___m.media-amazon.com_images_I_51m1gdQJW2L.jpg', 'https___m.media-amazon.com_images_I_71LRdXdqc0L.jpg', 'https___m.media-amazon.com_images_I_21mMXLWiDOL.jpg', 'https___m.media-amazon.com_images_I_61md5v6UPNL.jpg', 'https___m.media-amazon.com_images_I_71FMi9tO3HL.jpg', 'https___m.media-amazon.com_images_I_71brV+lqbRL.jpg', 'https___m.media-amazon.com_images_I_81zqul01R-L.jpg', 'https___m.media-amazon.com_images_I_91ozgHYE7sL.jpg', 'https___m.media-amazon.com_images_I_81bvW9AAggL.jpg', 'https___m.media-amazon.com_images_I_71gtFKX66JL.jpg']
'''

LABEL_NAME = "price"
IMAGE_FEATURE_NAME = "product_image"


def load_df(dir_path: str) -> DataFrame:
    csv_path = join(dir_path, "student_resource/dataset/train.csv")
    df = pd.read_csv(csv_path)
    img_folder = join(dir_path, IMAGE_FOLDER)
    df = download_url_image_column(df=df, img_folder=img_folder, img_col="image_link")
    df.rename(columns={"image_link": IMAGE_FEATURE_NAME}, inplace=True)
    df.drop(columns=["sample_id"], inplace=True)
    return df




CONTEXT = "Amazon ML Challenge 2025 - Price Prediction from Product Details"
TARGET = CuratedTarget(raw_name=LABEL_NAME, task_type=SupervisedTask.REGRESSION)
FEATURES = [CuratedFeature(raw_name=IMAGE_FEATURE_NAME, feat_type=FeatureType.IMAGE),
            CuratedFeature(raw_name="catalog_content", feat_type=FeatureType.TEXT)]
IMAGE_FOLDER = "self_downloaded_images"
LOADING_FUNC = load_df
