from os.path import exists, join
from typing import Optional

from pandas import DataFrame

from tabstar2.utils.datasets import load_csv
from tabstar_paper.datasets.curation_objects import CuratedTarget, CuratedFeature
from tabstar_paper.datasets.objects import SupervisedTask, FeatureType


'''
Dataset Name: deathtrooper/multichannel-glaucoma-benchmark-dataset/
====
Examples: 12449
====
URL: https://www.kaggle.com/deathtrooper/multichannel-glaucoma-benchmark-dataset
====
Description: 

SMDG, A Standardized Fundus Glaucoma Dataset
Standardized Multi-Channel Dataset for Glaucoma of 19 public datasets (SMDG-19)

About Dataset
Standardized Multi-Channel Dataset for Glaucoma (SMDG-19), a standardization of 19 public glaucoma datasets for AI applications.
Standardized Multi-Channel Dataset for Glaucoma (SMDG-19) is a collection and standardization of 19 public datasets, comprised of full-fundus glaucoma images, associated image metadata like, optic disc segmentation, optic cup segmentation, blood vessel segmentation, and any provided per-instance text metadata like sex and age. This dataset is designed to be exploratory and open-ended with multiple use cases and no established training/validation/test cases. This dataset is the largest public repository of fundus images with glaucoma.

Citation
Please cite at least the first work in academic publications:

Kiefer, Riley, et al. "A Catalog of Public Glaucoma Datasets for Machine Learning Applications: A detailed description and analysis of public glaucoma datasets available to machine learning engineers tackling glaucoma-related problems using retinal fundus images and OCT images." Proceedings of the 2023 7th International Conference on Information System and Data Mining. 2023.
R. Kiefer, M. Abid, M. R. Ardali, J. Steen and E. Amjadian, "Automated Fundus Image Standardization Using a Dynamic Global Foreground Threshold Algorithm," 2023 8th International Conference on Image, Vision and Computing (ICIVC), Dalian, China, 2023, pp. 460-465, doi: 10.1109/ICIVC58118.2023.10270429.
Kiefer, Riley, et al. "A Catalog of Public Glaucoma Datasets for Machine Learning Applications: A detailed description and analysis of public glaucoma datasets available to machine learning engineers tackling glaucoma-related problems using retinal fundus images and OCT images." Proceedings of the 2023 7th International Conference on Information System and Data Mining. 2023.
R. Kiefer, J. Steen, M. Abid, M. R. Ardali and E. Amjadian, "A Survey of Glaucoma Detection Algorithms using Fundus and OCT Images," 2022 IEEE 13th Annual Information Technology, Electronics and Mobile Communication Conference (IEMCON), Vancouver, BC, Canada, 2022, pp. 0191-0196, doi: 10.1109/IEMCON56893.2022.9946629.
Please also see the following optometry abstract publications:

A Comprehensive Survey of Publicly Available Glaucoma Datasets for Automated Glaucoma Detection; AAO 2022; https://aaopt.org/past-meeting-abstract-archives/?SortBy=ArticleYear&ArticleType=&ArticleYear=2022&Title=&Abstract=&Authors=&Affiliation=&PROGRAMNUMBER=225129
Standardized and Open-Access Glaucoma Dataset for Artificial Intelligence Applications; ARVO 2023; https://iovs.arvojournals.org/article.aspx?articleid=2790420
Ground truth validation of publicly available datasets utilized in artificial intelligence models for glaucoma detection; ARVO 2023; https://iovs.arvojournals.org/article.aspx?articleid=2791017
Please also see the DOI citations for this and related datasets:

SMDG; @dataset{smdg,
title={SMDG,
A Standardized Fundus Glaucoma Dataset},
url={https://www.kaggle.com/ds/2329670},
DOI={10.34740/KAGGLE/DS/2329670},
publisher={Kaggle},
author={Riley Kiefer},
year={2023}
}
EyePACS-light-v1 @dataset{eyepacs-light-v1,
title={Glaucoma Dataset: EyePACS AIROGS - Light},
url={https://www.kaggle.com/ds/3222646},
DOI={10.34740/KAGGLE/DS/3222646},
publisher={Kaggle},
author={Riley Kiefer},
year={2023}
}
EyePACS-light-v2 @dataset{eyepacs-light-v2,
title={Glaucoma Dataset: EyePACS-AIROGS-light-V2},
url={https://www.kaggle.com/dsv/7300206},
DOI={10.34740/KAGGLE/DSV/7300206},
publisher={Kaggle},
author={Riley Kiefer},
year={2023}
}
Dataset Objective
The objective of this dataset is a machine learning-ready dataset for glaucoma-related applications. Using the help of the community, new open-source glaucoma datasets will be reviewed for standardization and inclusion in this dataset.

Data Standardization
Full fundus images (and corresponding segmentation maps) are standardized using a novel algorithm (Citation 1) by cropping the background, centering the fundus image, padding missing information, and resizing to 512x512 pixels. This standardization ensures that the most amount of foreground information is prevalent during the resizing process for machine-learning-ready image processing.
Each available metadata text is standardized by provided each fundus image as a row and each fundus attribute as a column in a CSV file
Dataset Instance	Original Fundus	Standardized Fundus Image
sjchoi86-HRF		
BEH		
The following public glaucoma datasets are included in this multi-channel glaucoma benchmark dataset.

Dataset	0 (Non-Glaucoma)	1 (Glaucoma)	-1 (Glaucoma Suspect)
BEH (Bangladesh Eye Hospital)	463	171	0
CRFO-v4	31	48	0
DR-HAGIS (Diabetic Retinopathy, Hypertension, Age-related macular degeneration and Glacuoma ImageS)	0	10	0
DRISHTI-GS1-TRAIN	18	32	0
DRISHTI-GS1-TEST	13	38	0
EyePACS-AIROGS	0	3269	0
FIVES	200	200	0
G1020	724	296	0
HRF (High Resolution Fundus)	15	15	0
JSIEC-1000 (Joint Shantou International Eye Center)	38	0	13
LES-AV	11	11	0
OIA-ODIR-TRAIN	2932	197	18
OIA-ODIR-TEST-ONLINE	802	58	25
OIA-ODIR-TEST-OFFLINE	417	36	9
ORIGA-light	482	168	0
PAPILA	333	87	68
REFUGE1-TRAIN (Retinal Fundus Glaucoma Challenge 1 Train)	360	40	0
REFUGE1-VALIDATION (Retinal Fundus Glaucoma Challenge 1 Validation)	360	40	0
sjchoi86-HRF	300	101	0
Total	7499	4817	133
Instructions for Popular Use Cases
Glaucoma classification (12,449 total instances): Split the data by 'types' column in the CSV file. Input = 'fundus' file. Label = 'types' number.
Optic cup segmentation (2,874 instances): Find all rows in CSV file with a non-empty 'fundus_od_seg' column. Input = 'fundus' file. Label = 'fundus_oc_seg' file.
Optic disc segmentation (3,103 instances): Find all rows in CSV file with a non-empty 'fundus_oc_seg' column. Note some instances are labeled as 'Not Visible', so you must exclude these as well. Input = 'fundus' file. Label = 'fundus_od_seg' file.
Blood vessel segmentation (462 instances): Find all rows in CSV file with a non-empty 'bv_seg' column.
File Descriptions
metadata.csv : Links dataset instance metadata to image file paths.
full-fundus/ : Folder containing all full fundus images.
optic-cup/ : Folder containing the optic cup segmentation map based on the full fundus image.
optic-disc/ : Folder containing the optic disc segmentation map based on the full fundus image.
blood-vessel/ : Folder containing the blood vessel segmentation map based on the full fundus image.
vessel-artery/ : Folder containing the artery segmentation map based on the full fundus image.
vessel-vein/ : Folder containing the vein segmentation map based on the full fundus image.
spectral-oct/ : Folder containing all full spectral oct images.
spectral-oct-cup/ : Folder containing the optic cup segmentation lines based on the full spectral oct image.
spectral-oct-disc/ : Folder containing the optic disc segmentation lines based on the full spectral oct image.
Submit new datasets, suggest changes, and report bugs in this associated GitHub: https://github.com/TheBeastCoding/standardized-multichannel-dataset-glaucoma
====
Target Variable: types (int64, 3 distinct): ['0', '1', '-1']
====
Features:

fundus (object, 12449 distinct): ['full-fundus/full-fundus/OIA-ODIR-TEST-OFFLINE-1.png', 'full-fundus/full-fundus/OIA-ODIR-TEST-OFFLINE-2.png', 'full-fundus/full-fundus/OIA-ODIR-TEST-OFFLINE-4.png', 'full-fundus/full-fundus/OIA-ODIR-TEST-OFFLINE-5.png', 'full-fundus/full-fundus/OIA-ODIR-TEST-OFFLINE-6.png', 'full-fundus/full-fundus/OIA-ODIR-TEST-OFFLINE-7.png', 'full-fundus/full-fundus/OIA-ODIR-TEST-OFFLINE-8.png', 'full-fundus/full-fundus/OIA-ODIR-TEST-OFFLINE-9.png', 'full-fundus/full-fundus/OIA-ODIR-TEST-OFFLINE-10.png', 'full-fundus/full-fundus/OIA-ODIR-TEST-OFFLINE-11.png']
sex (object, 2 distinct, 63.7% missing): ['M', 'F']
age (float64, 77 distinct, 59.8% missing): ['56.0', '60.0', '55.0', '62.0', '54.0', '65.0', '64.0', '59.0', '63.0', '66.0']
eye (object, 2 distinct, 53.9% missing): ['OS', 'OD']
sbp (float64, 20 distinct, 99.8% missing): ['156.0', '153.0', '177.0', '167.0', '176.0', '127.0', '136.0', '162.0', '130.0', '123.0']
dbp (float64, 15 distinct, 99.8% missing): ['80.0', '79.0', '76.0', '81.0', '70.0', '95.0', '47.0', '64.0', '86.0', '90.0']
hr (float64, 16 distinct, 99.8% missing): ['68.0', '51.0', '66.0', '67.0', '63.0', '75.0', '52.0', '62.0', '84.0', '64.0']
iop (float64, 11 distinct, 99.8% missing): ['15.0', '13.0', '14.0', '8.0', '18.0', '12.0', '17.0', '16.0', '22.0', '19.0']
left_right (object, 2 distinct, 63.3% missing): ['left', 'right']
'''

def load_df(dir_path: str) -> DataFrame:
    df = load_csv(dir_path, "metadata - standardized.csv")
    df[FUNDUS] = df[FUNDUS].apply(lambda f: _fix_fundus_img_path(fundus=f, dir_path=dir_path))
    df['left_right'] = df['original_name'].apply(lambda o: get_left_right(o))
    bad_cols = [c for c in df.columns if 'Unnamed' in c]
    df = df.drop(columns=bad_cols)
    return df


def _fix_fundus_img_path(fundus: str, dir_path: str):
    assert fundus.count('/') == 2 and fundus.startswith('/')
    fundus = fundus[1:]
    fundus_name, file_name = fundus.split('/')
    new_path = join(fundus_name, fundus_name, file_name)
    assert exists(join(dir_path, new_path))
    return new_path


def get_left_right(org: str) -> Optional[str]:
    if 'left' in org.lower():
        return 'left'
    elif 'right' in org.lower():
        return 'right'
    else:
        return None


FUNDUS = "fundus"


CONTEXT = ""
TARGET = CuratedTarget(raw_name='types', task_type=SupervisedTask.MULTICLASS)
FULLY_MISSING_FIELS = ['Unnamed  24', 'vcdr', 
                       'notchI_present', 'notchS_present', 'notchN_present', 'notchT_present', 
                       'expert1_grade', 'expert2_grade', 'expert3_grade', 'expert4_grade', 'expert5_grade', 
                       'cdr_avg', 'cdr_expert1', 'cdr_expert2', 'cdr_expert3', 'cdr_expert4', 
                       'refractive_dioptre_1', 'refractive_dioptre_2', 'refractive_astigmatism', 
                       'phakic_or_pseudophakic', 
                       'iop_perkins', 'iop_pneumatic', 'pachymetry', 'axial_length', 'visual_field_mean_defect',
                       'type_expanded', 
                       'fundus_od_seg', 'fundus_oc_seg',
                       'oct', 'oct_od_seg', 'oct_oc_seg', 
                       'gender']
MOSTLY_MISSING = ['bv_seg', 'artery_seg', 'vein_seg']
COLS_TO_DROP = ['patient_id', 'isColor', 'names', 'original_name'] + FULLY_MISSING_FIELS + MOSTLY_MISSING
FEATURES = [CuratedFeature(raw_name=FUNDUS, feat_type=FeatureType.IMAGE)]
IMAGE_FOLDER = ""
LOADING_FUNC = load_df
