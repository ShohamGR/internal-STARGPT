from os.path import exists, join

from pandas import DataFrame

from tabstar2.utils.datasets import load_csv
from tabstar_paper.datasets.curation_objects import CuratedTarget, CuratedFeature
from tabstar_paper.datasets.objects import SupervisedTask, FeatureType


'''
Dataset Name: nguynththanhho/cmmd-mammography/
====
Examples: 5202
====
URL: https://www.kaggle.com/nguynththanhho/cmmd-mammography
====
Description: 

From CLIMB Paper:

CMMD (Cui et al., 2021) is a breast mammography dataset for 1,775 patients from China. Our evaluation utilizes the
diagnostic labels, ”Benign” and ”Malignant”, which are confirmed through biopsy. However, due to issues found
in the provided labels, while this dataset is a part of CLIMB, it is not a part of the experiment while pending expert
verifications.
Split: We use a training set of 1,404 records, a test set of 468 records, and a total size of 1,872 records.
Access restrictions: The dataset is available to download from https://www.cancerimagingarchive.net/collection/cmmd/.
Licenses: This dataset is available in the Creative Commons Attribution 4.0 International License
https://creativecommons.org/licenses/by/4.0/
Ethical considerations: No personally identifiable information or offensive content is present in the dataset
====
Target Variable: Classification (object, 2 distinct): ['Malignant', 'Benign']
====
Features:

LeftRight (object, 2 distinct): ['L', 'R']
Age (int64, 69 distinct): ['46', '44', '49', '43', '51', '50', '45', '42', '41', '47']
Abnormality (object, 3 distinct): ['mass', 'both', 'calcification']
Subtype (object, 4 distinct, 43.3% missing): ['Luminal B', 'Luminal A', 'HER2-enriched', 'triple negative']
Path (object, 5202 distinct): ['Cancer/D1-0033_1-2.png', 'Cancer/D1-0033_1-1.png', 'Cancer/D2-0230_1-2.png', 'Cancer/D2-0230_1-4.png', 'Cancer/D2-0230_1-3.png', 'Cancer/D2-0230_1-1.png', 'Cancer/D1-1575_1-2.png', 'Cancer/D1-1575_1-1.png', 'Cancer/D1-0889_1-2.png', 'Cancer/D1-0889_1-1.png']
Method_crop (object, 2 distinct): ['Yolo', 'Contour']
'''

PATH = "Path"


def load_df(dir_path: str) -> DataFrame:
    main_dir = join(dir_path, MAIN_DIR) 
    df = load_csv(main_dir, "description.csv")
    df[PATH] = df[PATH].apply(lambda path: validate_images(path, dir_path=dir_path))
    return df

def validate_images(pa: str, dir_path: str) -> str | None:
    if not exists(join(dir_path, IMAGE_FOLDER, pa)):
        return None
    return pa

CONTEXT = ""
TARGET = CuratedTarget(raw_name="Classification", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = ['ID']
FEATURES = [CuratedFeature(raw_name="Path", feat_type=FeatureType.IMAGE)]
MAIN_DIR = "CMMD"
IMAGE_FOLDER = MAIN_DIR
LOADING_FUNC = load_df
