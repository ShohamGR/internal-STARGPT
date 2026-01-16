import os
from os.path import exists, join

from pandas import DataFrame, read_csv

from tabstar2.utils.datasets import load_csv
from tabstar_paper.datasets.curation_objects import CuratedTarget, CuratedFeature
from tabstar_paper.datasets.objects import SupervisedTask, FeatureType


'''
Dataset Name: guardeec/mkphoto2023/
====
Examples: 13748
====
URL: https://www.kaggle.com/guardeec/mkphoto2023
====
Description:
title: MKPHOTO2023
subtitle: Profile photos of malicious social bots from VKontakte
keywords: ['russia', 'tabular', 'image', 'social networks']
licenses: [{'name': 'MIT'}]
description: The MKPHOTO2023 dataset is designed to research the types of profile photos used by different types of malicious social bots. This dataset includes photos employed by VKontakte bots, classification of photo types and bots' metrics. For this classification, we utilized various detectors:
1. [YOLO](https://ieeexplore.ieee.org/document/7780460) ([GitHub](https://github.com/WongKinYiu/yolov7)) - to identify a person
2. [CelebDetector](https://medium.com/analytics-vidhya/celebrity-recognition-using-vggface-and-annoy-363c5df31f1e) ([GitHub](https://github.com/shobhit9618/celeb_recognition)) - to identify face and celebrity
3. [GAN-image-detection](https://arxiv.org/pdf/2203.02246v1.pdf) ([GitHub](https://github.com/polimi-ispl/gan-image-detection)) - to identify GAN usage.
4. [DTM-image-detection](https://arxiv.org/pdf/2211.00680) ([GitHub](https://github.com/grip-unina/DMimageDetection)) - to identify Diffusion and Transformers models (DTM) usage.

GAN and DTM images were manually reviewed to clean up any misclassifications.

To collect bots and measure their metrics, we created 'honeypots' (fake victims) in VK and bought bot activity. During bot purchase, we measured bot [properties](https://www.researchgate.net/publication/368714138_Social_bot_metrics) (e.g. speed of action, price, etc.).

More details related to the photo analysis process: [preprint](https://www.researchgate.net/publication/377230758_The_Face_of_Deception_The_Impact_of_AI-Generated_Photos_on_Malicious_Social_Bots)

More details related to bot purchase and bots' metrics measurements process: [paper](https://link.springer.com/article/10.1007/s13278-023-01038-3), [preprint](https://www.researchgate.net/publication/368714138_Social_bot_metrics)

___

The dataset consists of the following files:

1. **photos.zip** - is the archive with .JPG photos of bots' faces. Each photo has an **id** as the name of the file.
2. **dataset.csv** - is a result of photo analysis where we aggregated outputs of various detectors (see files below) and added bots' metrics from [MKMETRIC2022](https://github.com/guardeec/datasets/tree/main#mkmetric2022) dataset.

Additional files below are the raw output of detectors.

3. **celebs_and_faces.csv** - is the raw output of CelebDetector.
4. **face_labels.csv** - is the raw output of YOLO detector.
5. **gan.csv** - is the raw output of GAN detector.
6. **dtm.csv** - is the raw output of DTM detector.
====
Target Variable: TrustNZ_IDNNZ (float64, 52 distinct): ['0.4915', '0.3913', '0.5', '0.5217', '0.5385', '0.541', '0.55', '0.3793', '0.3', '0.4783']
====
Features:

gan (int64, 2 distinct): ['0', '1']
dtm (int64, 2 distinct): ['0', '1']
person (float64, 2 distinct, 7.7% missing): ['1.0', '0.0']
n_faces (int64, 15 distinct): ['1', '-1', '2', '3', '4', '5', '9', '6', '25', '50']
celebs (object, 32 distinct): ['[]', "['kanganaranaut']", "['gururandhawa']", "['leonardodicaprio']", "['vickykaushal']", "['vijaydevarakonda']", "['aliabhatt']", "['edsheeran']", "['nickjonas']", "['priyankachopra']"]
dataset (object, 65 distinct): ['followers.txt', 'Z1Y1X1_A.txt', 'smmOG_B.txt', 'smmOG_A.txt', 'MARSHRUTKA_HIGH.txt', 'Z1Y1X1_B.txt', 'Hideflow_C.txt', 'Cashbox.txt', 'tmsmm _B.txt', 'Likeinsta.txt']
SR (float64, 63 distinct): ['0.0328', '0.3429', '0.0161', '0.0476', '0.0938', '0.6721', '0.0612', '0.0294', '0.1296', '0.2716']
price (float64, 45 distinct): ['1.0', '0.15', '0.035', '0.3', '0.09', '0.34', '0.13', '0.54', '0.21', '0.52']
BTT (float64, 2 distinct): ['0.0', '1.0']
speed (float64, 3 distinct): ['1.0', '2.0', '0.0']
NBQ (float64, 3 distinct): ['2.0', '0.0', '1.0']
image (object, 11423 distinct): ['301503357.jpg', '21691950.jpg', '460856719.jpg', '591795381.jpg', '634958642.jpg', '683208262.jpg', '655005745.jpg', '500200177.jpg', '307676215.jpg', '464818115.jpg']
'''

IMAGE_FEATURE_NAME = "image"

def load_df(dir_path: str) -> DataFrame:
    df = load_csv(dir_path, "dataset.csv")
    df[IMAGE_FEATURE_NAME] = df['id'].apply(lambda img: get_img(img, dir_path=dir_path))    # celebs_and_faces.csv  dataset.csv  dtm.csv  face_labels.csv  faces_400  gan.csv
    return df


def get_img(img: str, dir_path: str) -> str:
    img_folder = join(dir_path, IMAGE_FOLDER)
    img_filename = f"{img}.jpg"
    assert exists(join(img_folder, img_filename))
    return img_filename


CONTEXT = ""
TARGET = CuratedTarget(raw_name='TrustNZ_IDNNZ', task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = ['id']
FEATURES = [CuratedFeature(raw_name=IMAGE_FEATURE_NAME, feat_type=FeatureType.IMAGE)]
IMAGE_FOLDER = "faces_400/faces_400"
LOADING_FUNC = load_df
