from os.path import exists, join

from pandas import DataFrame

from tabstar2.utils.datasets import load_csv
from tabstar_paper.datasets.curation_objects import CuratedTarget, CuratedFeature
from tabstar_paper.datasets.objects import SupervisedTask, FeatureType


'''
Dataset Name: andreacombette/pinterest-analysis-using-nlp-and-image-analysis/
====
Examples: 1504
====
URL: https://www.kaggle.com/andreacombette/pinterest-analysis-using-nlp-and-image-analysis
====
Description: 
About Dataset
Data Science Applications

The dataset's structure and content make it ideal for a variety of data science applications, including:

Content Analysis: Explore the themes and subjects in the descriptions and titles of the most popular pins.
Popularity Metrics: Utilize repin counts to measure content virality and audience interest.
Trend Identification: Identify trending topics and styles among top Pinterest influencers.
NLP (Natural Language Processing): Analyze textual data for sentiment analysis, keyword extraction, and trend prediction.
-Image Analysis : Analyse Images distance, through multiple metrics, Vision Algorithm
graph Analysis : For clustering and features extraction
Column Descriptors
The dataset is concise yet informative, comprising the following columns:

-ID: A unique identifier for each pin, facilitating easy reference and analysis.
-Description: The textual description provided for each pin, offering insights into the content and its appeal.
-Title: The title of the pin, which may contain key information or keywords relevant to the content.
-Repin Count: A quantitative measure of the pin's popularity, indicating how often it has been repinned by users.

Graph description :
The nxGraph is composed of following edges metrics :

Description similarities : Compute with gensim : word2vec-google-news-300 model.
Title similarities : Compute with gensim : word2vec-google-news-300 model.
For the Nodes metrics :

Metrics from datasets
Images Metrics (from sklearn.stats.describe) for each image
Images the images correspond to pin id of the dataset. They are resized to 64x64 pixels.

Acknowledgements

We are thankful to the vibrant community of Pinterest and its top contributors whose creativity and engagement have made this dataset possible. Their dedication to sharing and curating content has offered us a window into the dynamics of social media engagement and content popularity.

Ethically Mined Data
This dataset upholds the highest standards of ethical data collection. It has been compiled with respect for user privacy and in alignment with Pinterest's data usage policies. By focusing on publicly available data such as pin descriptions and repin counts, the dataset ensures respect for individual privacy while providing valuable insights for analysis.

Thanks to Oneli WICKRAMASINGHE for releasing this dataset
====
Target Variable: is_repin (int64, 2 distinct): ['1', '0']
====
Features:

description (object, 1226 distinct): [' ', "Modern garden vibes made this Maui wedding one for the books! With all-white fashion and adorable details to match the breathtaking venue's natural features, this wedding was sheer perfection! See the link in our bio for an even deeper look!", 'The rain at this Colorado venue only added to the romance of this Bridgerton-inspired wedding! For every little detail, see the link in our bio!', 'This wedding was a vision of soft elegance, where a muted yet varied color palette set the tone for a celebration that felt like a dream. Blushes, pinks, corals, creams, greens, and blues blended seamlessly, creating a harmonious canvas that mirrored the serene beauty of Tahoe.', 'This stunning coastal California soiree blended beautiful hues of orange, blue, and green, resulting in an unforgettable palette for an equally unforgettable wedding! See the link in our bio for the full gallery!', "If you've ever wondered what a Provence wedding looks like, look no further than this sun-drenched fete! Lush sorbet-toned florals added to the inspired ambiance, along with other cute, noteworthy details! Which detail stands out to you the most? See the link in our bio for the full gallery! Little Black Book Photographer: @thomasaudiffren", 'from @urbanoutfitters', 'This Los Angeles wedding is the blueprint for blending old world charm with modern whimsy! What detail stands out to you the most? See the link in our bio for an even deeper look!', 'This southern wedding was filled with timeless touches. See every beautiful moment at the link in our bio! Little Black Book Venue: @hewittoaks Little Black Book Photographer: @kaitlyndelongphotography Little Black Book Event Planning & Floral', 'This style-centric vineyard wedding wowed its guests with the most stunning balance of whites and greens! From the playful parasols to the chic tablescape, see the link in our bio for more classic wedding inspo!']
title (object, 1404 distinct): ['https://www.stylemepretty.com/2023/12/29/a-maui-destination-wedding-with-modern-garden-vibes/', 'Al Fresco Wedding on the French Riviera at Bastide du Roy', 'California Sunshine Coastal Soiree With Bright Bursts Of Orange', 'Romantic Bridgerton-Inspired Wedding With a Modern Twist', 'Quintessential Provence Wedding at Le Galinier de Lourmarin', 'Timeless Southern Garden-Inspired Wedding In South Carolina', 'Sunshine and Storm Clouds Made for the Most Magical Tahoe Wedding Day!', 'A Tented Summer Wedding at Dormie Club in a Neutral Palette', 'Dancing Under Spring Blooms in a Pastel Palette!', 'The Blueprint for Blending Old World Charm With Modern Whimsy at Vibiana in LA']
pinterest_img (object, 1504 distinct): ['21181060741360356.jpg', '21181060741360355.jpg', '21181060741360352.jpg', '21181060741360343.jpg', '21181060741360342.jpg', '21181060741360341.jpg', '21181060741341149.jpg', '21181060741267132.jpg', '21181060741267125.jpg', '21181060741267119.jpg']
'''

IS_REPIN = "is_repin"
PINTEREST_IMG = "pinterest_img"

def load_df(dir_path: str) -> DataFrame:
    df = load_csv(dir_path, "pinterest_finalised_2.csv")
    df[PINTEREST_IMG] = df["id"].apply(lambda x: collect_img(x, dir_path))
    df = repin_count_discretize(df)
    return df


def collect_img(pid: str, dir_path: str) -> str:
    img_path = f"{pid}.jpg"
    full_path = join(dir_path, IMAGE_FOLDER, img_path)
    if not exists(full_path):
        raise FileNotFoundError(f"Image not found: {full_path}")
    return img_path


def repin_count_discretize(df: DataFrame) -> DataFrame:
    df[IS_REPIN] = df['repin_count'].apply(lambda x: 1 if x > 0 else 0)
    df.drop(columns=['repin_count'], inplace=True)
    return df




CONTEXT = ""
TARGET = CuratedTarget(raw_name=IS_REPIN, task_type=SupervisedTask.BINARY)
COLS_TO_DROP = ['Unnamed: 0', 'id']
FEATURES = [CuratedFeature(raw_name=PINTEREST_IMG, feat_type=FeatureType.IMAGE),
            CuratedFeature(raw_name="description", feat_type=FeatureType.TEXT),
            CuratedFeature(raw_name="title", feat_type=FeatureType.TEXT)]
IMAGE_FOLDER = "images/images"
LOADING_FUNC = load_df
