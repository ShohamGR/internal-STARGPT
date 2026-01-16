import os
from os.path import join

from pandas import DataFrame, read_csv

from tabstar2.utils.datasets import load_csv
from tabstar_paper.datasets.curation_objects import CuratedTarget, CuratedFeature
from tabstar_paper.datasets.objects import SupervisedTask, FeatureType

IMAGE_FEATURE_NAME = "img_id"


'''
Dataset Name: mahdavi1202/skin-cancer/
====
Examples: 2298
====
URL: https://www.kaggle.com/mahdavi1202/skin-cancer
====
Description:
About Dataset
About Dataset

Summary description
Published: 7 Jul 2020

The skin lesions are: Basal Cell Carcinoma (BCC), Squamous Cell Carcinoma (SCC), Actinic Keratosis (ACK), Seborrheic Keratosis (SEK), Bowen’s disease (BOD), Melanoma (MEL), and Nevus (NEV). As the Bowen’s disease is considered SCC in situ, we clustered them together, which results in six skin lesions in the dataset, three skin cancers (BCC, MEL, and SCC) and three skin disease (ACK, NEV, and SEK)
All BCC, SCC, and MEL are biopsy-proven. The remaining ones may have clinical diagnosis according to a consensus of a group of dermatologists. In total, approximately 58% of the samples in this dataset are biopsy-proven. This information is described in the metadata.
The images present in the dataset have different sizes because they are collected using different smartphone devices. All images are available in .png format.
The metadata associated with each skin lesion is composed of up to 26 features. All features are available in a CSV document in which each line represents a skin lesion and each column a metadata feature.
In total, there are 1,373 patients, 1,641 skin lesions, and 2,298 images present in the dataset. Each image/sample has a reference to the patient and the skin lesion in the metadata.
Number of instances: 2298

Number of attributes: 26

1) patient_id: Identifier of the patient under study.

2) lesion_id: Identifier of the lesion or wound under study in the patient.

3) smoke: Whether the patient has a history of smoking or not.

4) drink: Whether the patient has a history of alcohol consumption or not.

5) background_father: The history of any diseases or health conditions related to the patient's father, including any history of skin cancer or other diseases that may be related to skin cancer.

6) background_mother: The history of any diseases or health conditions related to the patient's mother, including any history of skin cancer or other diseases that may be related to skin cancer.

7) age: Age of the patient at the time of examination.

8) pesticide: Whether the patient has been exposed to pesticides or other chemicals.

9) gender: Gender of the patient.

10) skin_cancer_history: History of skin cancer in the patient's family.

11) cancer_history: History of cancer in the patient's family.

12) has_piped_water: Indicates whether the location or area of the patient's residence has access to piped water or not.

13) has_sewage_system: Indicates whether the location or area of the patient's residence has a proper sewage system or not.

14) fitspatrick: Skin tolerance to sunlight.

15) region: The area of the body where the lesion or wound has been examined.

16) diameter_1: Primary diameter of the lesion or wound.

17) diameter_2: Secondary diameter of the lesion or wound.

18) diagnostic: The type of lesion or wound is diagnosed.

19) itch: Whether the lesion or wound has itched or not.

20) grew: Whether the size of the lesion or wound has grown or not.

21) hurt: Whether the lesion or wound has hurt or not.

22) changed: Whether the appearance of the lesion or wound has changed or not.

23) bleed: Whether the lesion or wound has bled or not.

24) elevation: Description of the of the lesion or wound relative to the skin surface of the patient.

25) img_id: Identifier of the image related to the lesion or wound.

26) biopsed: Whether the lesion or wound has been biopsied or not.


From MultiModalPFN Paper:
PAD–UFES–20 The PAD–UFES–20(Pacheco et al., 2020) dataset contains 2,298 samples of
six skin lesion types, each paired with a clinical image and up to 26 metadata features such as
age, lesion location, and diameter. Three lesion types correspond to cancers (Basal Cell Carcinoma, Squamous Cell Carcinoma including Bowen’s disease, and Melanoma), while the remaining three are noncancerous (Actinic Keratosis, Nevus, and Seborrheic Keratosis), with approximately 58% of samples biopsy-proven. The images are provided in .png format and were collected
using different smartphones, thereby resulting in varying image sizes that require preprocessing.
https://data.mendeley.com/datasets/zr7vgbcyr2/1
====
Target Variable: diagnostic (object, 6 distinct): ['BCC', 'ACK', 'NEV', 'SEK', 'SCC', 'MEL']
====
Features:

smoke (object, 2 distinct, 35.0% missing): ['0', '1']
drink (object, 2 distinct, 35.0% missing): ['0', '1']
background_father (object, 13 distinct, 35.6% missing): ['POMERANIA', 'GERMANY', 'ITALY', 'UNK', 'BRAZIL', 'NETHERLANDS', 'PORTUGAL', 'POLAND', 'BRASIL', 'CZECH']
background_mother (object, 11 distinct, 35.8% missing): ['POMERANIA', 'GERMANY', 'ITALY', 'UNK', 'BRAZIL', 'NETHERLANDS', 'PORTUGAL', 'POLAND', 'NORWAY', 'FRANCE']
age (int64, 84 distinct): ['55', '73', '58', '71', '62', '75', '57', '65', '53', '66']
pesticide (object, 2 distinct, 35.0% missing): ['0', '1']
gender (object, 2 distinct, 35.0% missing): ['FEMALE', 'MALE']
skin_cancer_history (object, 2 distinct, 35.0% missing): ['0', '1']
cancer_history (object, 2 distinct, 35.0% missing): ['1', '0']
has_piped_water (object, 2 distinct, 35.0% missing): ['1', '0']
has_sewage_system (object, 2 distinct, 35.0% missing): ['1', '0']
fitspatrick (float64, 6 distinct, 35.0% missing): ['2.0', '3.0', '1.0', '4.0', '5.0', '6.0']
region (object, 14 distinct): ['FACE', 'FOREARM', 'CHEST', 'BACK', 'ARM', 'NOSE', 'HAND', 'NECK', 'EAR', 'THIGH']
diameter_1 (float64, 42 distinct, 35.0% missing): ['10.0', '15.0', '7.0', '5.0', '6.0', '9.0', '8.0', '12.0', '11.0', '13.0']
diameter_2 (float64, 38 distinct, 35.0% missing): ['5.0', '10.0', '6.0', '7.0', '8.0', '4.0', '3.0', '9.0', '15.0', '12.0']
itch (object, 3 distinct): ['TRUE', 'FALSE', 'UNK']
grew (object, 3 distinct): ['FALSE', 'TRUE', 'UNK']
hurt (object, 3 distinct): ['FALSE', 'TRUE', 'UNK']
changed (object, 3 distinct): ['FALSE', 'UNK', 'TRUE']
bleed (object, 3 distinct): ['FALSE', 'TRUE', 'UNK']
elevation (object, 3 distinct): ['TRUE', 'FALSE', 'UNK']
img_id (object, 2298 distinct): ['imgs_part_3/imgs_part_3/PAT_1516_1765_530.png', 'imgs_part_1/imgs_part_1/PAT_46_881_939.png', 'imgs_part_3/imgs_part_3/PAT_1545_1867_547.png', 'imgs_part_3/imgs_part_3/PAT_1989_4061_934.png', 'imgs_part_2/imgs_part_2/PAT_684_1302_588.png', 'imgs_part_3/imgs_part_3/PAT_1549_1882_230.png', 'imgs_part_2/imgs_part_2/PAT_778_1471_835.png', 'imgs_part_1/imgs_part_1/PAT_117_179_983.png', 'imgs_part_3/imgs_part_3/PAT_1995_4080_695.png', 'imgs_part_2/imgs_part_2/PAT_705_4015_413.png']
biopsed (bool, 2 distinct): ['1', '0']
'''


def load_df(dir_path: str) -> DataFrame:
    df = load_csv(dir_path, "metadata.csv")
    df = find_exact_img_path(df=df, dir_path=dir_path)
    return df

def find_exact_img_path(df: DataFrame, dir_path: str):
    img2full = {}
    for n in [1, 2, 3]:
        img_folder = f"imgs_part_{n}/imgs_part_{n}"
        for f in os.listdir(join(dir_path, img_folder)):
            img2full[f] = join(img_folder, f)
    df[IMAGE_FEATURE_NAME] = df[IMAGE_FEATURE_NAME].map(img2full)
    return df



CONTEXT = ""
TARGET = CuratedTarget(raw_name="diagnostic", task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = ['patient_id', 'lesion_id']
FEATURES = [CuratedFeature(raw_name=IMAGE_FEATURE_NAME, feat_type=FeatureType.IMAGE)]
IMAGE_FOLDER = ""
LOADING_FUNC = load_df
