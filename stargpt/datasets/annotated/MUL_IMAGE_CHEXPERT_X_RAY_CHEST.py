from os.path import exists,join

from pandas import concat, DataFrame

from tabstar2.utils.datasets import load_csv
from tabstar_paper.datasets.curation_objects import CuratedTarget, CuratedFeature
from tabstar_paper.datasets.objects import SupervisedTask, FeatureType

X_RAY = "X-Ray Image"

'''
Dataset Name: ashery/chexpert/
====
Examples: 46437
====
URL: https://www.kaggle.com/ashery/chexpert
====
Description: 
About Dataset
This dataset is a smaller, downsampled version of the original dataset, which can be found here. It includes 224,316 chest radiographs from 65,240 patients, featuring both frontal and lateral views. The dataset is designed to aid in the automated interpretation of chest x-rays and includes uncertainty labels and evaluation sets annotated by radiologists. For details on how to reproduce the original CheXpert results, refer to the paper and this repository.
====
Target Variable: Cardiomegaly (float64, 3 distinct): ['1.0', '0.0', '-1.0']
====
Features:

Sex (object, 2 distinct): ['Male', 'Female']
Age (int64, 74 distinct): ['90', '61', '60', '63', '66', '64', '62', '56', '58', '68']
Frontal/Lateral (object, 2 distinct): ['Frontal', 'Lateral']
AP/PA (object, 3 distinct, 17.8% missing): ['AP', 'PA', 'LL']
No Finding (float64, 2 distinct, 92.3% missing): ['1.0', '0.0']
Enlarged Cardiomediastinum (float64, 3 distinct, 87.6% missing): ['1.0', '0.0', '-1.0']
Lung Opacity (float64, 3 distinct, 51.7% missing): ['1.0', '0.0', '-1.0']
Lung Lesion (float64, 3 distinct, 95.0% missing): ['1.0', '0.0', '-1.0']
Edema (float64, 3 distinct, 44.3% missing): ['1.0', '0.0', '-1.0']
Consolidation (float64, 3 distinct, 63.8% missing): ['0.0', '-1.0', '1.0']
Pneumonia (float64, 3 distinct, 89.3% missing): ['-1.0', '1.0', '0.0']
Atelectasis (float64, 3 distinct, 70.8% missing): ['-1.0', '1.0', '0.0']
Pneumothorax (float64, 3 distinct, 71.3% missing): ['0.0', '1.0', '-1.0']
Pleural Effusion (float64, 3 distinct, 34.1% missing): ['1.0', '0.0', '-1.0']
Pleural Other (float64, 3 distinct, 96.5% missing): ['1.0', '-1.0', '0.0']
Fracture (float64, 3 distinct, 94.7% missing): ['1.0', '0.0', '-1.0']
Support Devices (float64, 3 distinct, 46.2% missing): ['1.0', '0.0', '-1.0']
X Ray Image (object, 46437 distinct): ['train/patient00002/study2/view1_frontal.jpg', 'train/patient00005/study1/view1_frontal.jpg', 'train/patient00005/study1/view2_lateral.jpg', 'train/patient00007/study1/view1_frontal.jpg', 'train/patient00009/study1/view1_frontal.jpg', 'train/patient00009/study1/view2_lateral.jpg', 'train/patient00011/study12/view1_frontal.jpg', 'train/patient00012/study2/view1_frontal.jpg', 'train/patient00012/study2/view2_lateral.jpg', 'train/patient00017/study1/view1_frontal.jpg']
'''

def load_df(dir_path: str) -> DataFrame:
    train_df = load_csv(dir_path, "train.csv")
    valid_df = load_csv(dir_path, "valid.csv")
    df = concat([train_df, valid_df])
    df[X_RAY] = df['Path'].apply(lambda x: consolidate_img_path(x, dir_path=dir_path))
    return df


def consolidate_img_path(path: str, dir_path: str) -> str:
    prefix = 'CheXpert-v1.0-small/'
    if not path.startswith(prefix):
        raise ValueError(f"Path {path} does not start with {prefix}")
    path = path.replace(prefix, '')
    full_path = join(dir_path, IMAGE_FOLDER, path)
    if not exists(full_path):
        raise FileNotFoundError(f"Image not found: {full_path}")
    return path



CONTEXT = ""
TARGET = CuratedTarget(raw_name='Cardiomegaly', task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = [
                # Image path
                'Path']
FEATURES = [CuratedFeature(raw_name=X_RAY, feat_type=FeatureType.IMAGE)]
IMAGE_FOLDER = ""
LOADING_FUNC = load_df

