from os.path import join

from pandas import DataFrame, read_csv

from tabstar_paper.datasets.curation_objects import CuratedTarget, CuratedFeature
from tabstar_paper.datasets.objects import SupervisedTask, FeatureType

MAIN_FOLDER = "memotion_dataset_7k"

LABEL_NAME = "humour"
IMAGE_FEATURE_NAME = "image_name"

'''
Dataset Name: moumitanag2264/memotion-dataset/
====
Examples: 6992
====
URL: https://www.kaggle.com/moumitanag2264/memotion-dataset
====
Description:
====
Target Variable: humour (object, 4 distinct): ['funny', 'very_funny', 'not_funny', 'hilarious']
====
Features:

image_name (object, 6992 distinct): ['image_1.jpg', 'image_2.jpeg', 'image_3.JPG', 'image_4.png', 'image_5.png', 'image_6.jpg', 'image_7.png', 'image_8.jpg', 'image_9.jpg', 'image_10.png']
text_corrected (object, 6939 distinct): ['<html><head><meta content="text/html; charset=UTF-8" http-equiv="content-type"><style type="text/css">ol{margin:0;padding:0}table td', 'FEMINISM. YOU KEEP USING THAT WORD I DO NOT THINK IT MEANS WHAT YOU THINK IT MEANS memegenerator.net', "BUT I'M SUCH A NICE GUY memegenerator.net", 'Bill is on the internet. Bill sees something that offends him. Bill moves on. Bill is smart. Be like Bill.', 'THAT MOMENT WHEN YOU FIND THE PERFECT AVOCADO AT THE SUPERMARKET', "CAPTAIN'S LOG", 'OKAY', "You didn't cry when Bambi's mother died? Yes  it was very sad when the guy stopped drawing the deer.", "When you're trying to understand the chronology of X-Men films I'M HERE TO TALK TO YOU ABOUT THE AVENG GO FUCK YOURSELF", 'WITH THIS TECHNOLOGY WE WILL BRING THE UNITED STATES TO ITS KNEES']
'''

def load_df(dir_path: str) -> DataFrame:
    main_dir = join(dir_path, MAIN_FOLDER)
    df = read_csv(join(main_dir, "labels.csv"))
    df = df[[IMAGE_FEATURE_NAME, LABEL_NAME, "text_corrected"]]
    return df


CONTEXT = ""
TARGET = CuratedTarget(raw_name=LABEL_NAME, task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name=IMAGE_FEATURE_NAME, feat_type=FeatureType.IMAGE)]
IMAGE_FOLDER = f"{MAIN_FOLDER}/images"
LOADING_FUNC = load_df
