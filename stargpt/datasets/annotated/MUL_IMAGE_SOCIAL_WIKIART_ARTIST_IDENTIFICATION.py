from datasets import DatasetDict
from pandas import DataFrame

from tabstar_paper.datasets.curation_objects import CuratedTarget, CuratedFeature
from tabstar_paper.datasets.objects import SupervisedTask, FeatureType

LABEL_NAME = "artist"
IMAGE_FEATURE_NAME = "image"

'''
Dataset Name: MUL_IMAGE_SOCIAL_WIKIART_ARTIST_IDENTIFICATION
====
Examples: 81444
====
URL: https://huggingface.co/datasets/huggan/wikiart
====
Description: 
Dataset Summary
Dataset containing 81,444 pieces of visual art from various artists, taken from WikiArt.org, along with class labels for each image :

"artist" : 129 artist classes, including a "Unknown Artist" class
"genre" : 11 genre classes, including a "Unknown Genre" class
"style" : 27 style classes
On WikiArt.org, the description for the "Artworks by Genre" page reads :

A genre system divides artworks according to depicted themes and objects. A classical hierarchy of genres was developed in European culture by the 17th century. It ranked genres in high – history painting and portrait, - and low – genre painting, landscape and still life. This hierarchy was based on the notion of man as the measure of all things. Landscape and still life were the lowest because they did not involve human subject matter. History was highest because it dealt with the noblest events of humanity. Genre system is not so much relevant for a contemporary art; there are just two genre definitions that are usually applied to it: abstract or figurative.

The "Artworks by Style" page reads :

A style of an artwork refers to its distinctive visual elements, techniques and methods. It usually corresponds with an art movement or a school (group) that its author is associated with.

Dataset Structure
"image" : image
"artist" : 129 artist classes, including a "Unknown Artist" class
"genre" : 11 genre classes, including a "Unknown Genre" class
"style" : 27 style classes
Source Data
Files taken from this archive, curated from the WikiArt website.

Additional Information
Note:

The WikiArt dataset can be used only for non-commercial research purpose.
The images in the WikiArt dataset were obtained from WikiArt.org.
The authors are neither responsible for the content nor the meaning of these images.
By using the WikiArt dataset, you agree to obey the terms and conditions of WikiArt.org.
Contributions
gigant added this dataset to the hub.
====
Target Variable: artist (object, 129 distinct): ['Unknown Artist', 'vincent-van-gogh', 'nicholas-ro]
====
Features:

image (image path): Image dicts, with keys 'bytes' and 'path'
genre (object, 11 distinct): ['Unknown Genre', 'portrait', 'landscape', 'genre_painting', 'religiou]
style (object, 27 distinct): ['Impressionism', 'Realism', 'Romanticism', 'Expressionism', 'Post_Imp]
'''

def load_df(hf_dataset: DatasetDict) -> DataFrame:
    df = load_wikiart_df(hf_dataset)
    df = take_top_10_most_common_artists(df)
    return df

def load_wikiart_df(hf_dataset: DatasetDict) -> DataFrame:
    df = hf_dataset["train"].to_pandas()
    for col in df.columns:
        if col == IMAGE_FEATURE_NAME:
            continue
        labels = hf_dataset['train'].features[col].names
        df[col] = df[col].apply(lambda x: labels[x])
    return df

def take_top_10_most_common_artists(df: DataFrame) -> DataFrame:
    artist_counts = df[LABEL_NAME].value_counts()
    top_10_artists = artist_counts.head(10).index
    df = df[df[LABEL_NAME].isin(top_10_artists)]
    return df

CONTEXT = "Artist identification from social media images and metadata"
TARGET = CuratedTarget(raw_name=LABEL_NAME, task_type=SupervisedTask.MULTICLASS)
FEATURES = [CuratedFeature(raw_name=IMAGE_FEATURE_NAME, feat_type=FeatureType.IMAGE)]
LOADING_FUNC = load_df
