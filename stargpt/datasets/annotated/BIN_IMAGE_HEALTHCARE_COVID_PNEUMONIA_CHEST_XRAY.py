from os.path import join

from pandas import DataFrame, read_csv

from tabstar_paper.datasets.curation_objects import CuratedTarget, CuratedFeature
from tabstar_paper.datasets.objects import SupervisedTask, FeatureType

'''
Dataset Name: bachrr/covid-chest-xray/
====
Examples: 372
====
URL: https://www.kaggle.com/bachrr/covid-chest-xray
====
Description: 
COVID-19 X-ray images
About
This dataset is a database of COVID-19 cases with chest X-ray or CT images. It contains COVID-19 cases as well as MERS, SARS, and ARDS.

Background
COVID is possibly better diagnosed using radiological imaging Fang, 2020. Companies are developing AI tools and deploying them at hospitals Wired 2020. We should have an open database to develop free tools that will also provide assistance.

Contribute
Your help is needed, use these images in Kaggle kernels to develop AI-based approaches to predict and understand COVID-19. To learn more about the dataset visit the GitHub repo - covid-chestxray-dataset.

Metadata
Here is a list of each metadata field, with explanations:

Patientid (internal identifier, just for this dataset)
offset (number of days since the start of symptoms or hospitalization for each image, this is very important to have when there are multiple images for the same patient to track progression while being imaged. If a report says "after a few days" let's assume 5 days.)
sex (M, F, or blank)
age (age of the patient in years)
finding (which pneumonia)
survival (did they survive? Y or N)
view (for example, PA, AP, or L for X-rays and Axial or Coronal for CT scans)
modality (CT, X-ray, or something else)
date (date the image was acquired)
location (hospital name, city, state, country) importance from right to left.
filename
doi (DOI of the research article
url (URL of the paper or website where the image came from)
license
clinical notes (about the radiograph in particular, not just the patient)
other notes (e.g. credit)
====
Target Variable: finding (object, 11 distinct): ['COVID-19', 'Streptococcus', 'SARS', 'Pneumocystis', 'COVID-19, ARDS', 'ARDS', 'E.Coli', 'No Fin]
====
Features:

patientid (int64, 204 distinct): ['19', '13', '205', '17', '178', '173', '117', '132', '51', '31']
offset (float64, 27 distinct): ['0.0', '5.0', '3.0', '7.0', '10.0', '2.0', '4.0', '1.0', '9.0', '6.0']
sex (object, 2 distinct): ['M', 'F']
age (float64, 55 distinct): ['70.0', '50.0', '55.0', '65.0', '60.0', '73.0', '35.0', '40.0', '45.0', '75.0']
survival (object, 2 distinct): ['Y', 'N']
intubated (object, 2 distinct): ['Y', 'N']
intubation_present (object, 2 distinct): ['N', 'Y']
went_icu (object, 2 distinct): ['Y', 'N']
in_icu (object, 2 distinct): ['Y', 'N']
needed_supplemental_O2 (object, 2 distinct): ['Y', 'N']
extubated (object, 2 distinct): ['Y', 'N']
temperature (float64, 20 distinct): ['38.0', '39.0', '38.9', '37.8', '38.2', '37.5', '39.6', '38.6', '36.4', '37.2']
pO2_saturation (float64, 18 distinct): ['97.0', '98.0', '96.0', '92.0', '89.0', '93.0', '84.0', '85.0', '95.0', '70.0']
leukocyte_count (float64, 11 distinct): ['7.4', '2.88', '6.84', '6.37', '11.2', '3.13', '5.5', '3.15', '6.91', '6.4']
neutrophil_count (float64, 2 distinct): ['1.63', '5.55']
lymphocyte_count (float64, 9 distinct): ['0.8', '0.9', '1.2', '1.3', '1.73', '0.63', '0.6', '0.7', '0.131']
view (object, 7 distinct): ['PA', 'AP', 'AP Supine', 'Axial', 'L', 'Coronal', 'AP semi erect']
modality (object, 2 distinct): ['X-ray', 'CT']
date (object, 63 distinct): ['2020', '2004', '2016', '2014', 'February 6, 2020', '2018', 'Mar 4, 2020', '2015', '2011', 'Mar 3, 2020']
location (object, 55 distinct): ['Italy', 'Spain', 'Mount Sinai Hospital, Toronto, Ontario, Canada', 'Wenzhou, China', 'Taoyuan General Hospital,]
filename (object, 372 distinct): ['auntminnie-a-2020_01_28_23_51_6665_2020_01_28_Vietnam_coronavirus.jpeg', 'auntminnie-b-2020_01_28_23_51_6665_2]
# clinical_notes (object, 287 distinct): ['Small consolidation in right upper lobe and ground-glass opacities in both lower lobes were observed on ]
'''

LABEL_NAME = "finding"
IMAGE_FEATURE_NAME = "filename"


def load_df(dir_path: str) -> DataFrame:
    df_path = join(dir_path, "metadata.csv")
    df = read_csv(df_path)
    df[LABEL_NAME] = df[LABEL_NAME].apply(is_covid)
    return df

def is_covid(finding: str) -> str:
    '''
    finding
    COVID-19          296
    Streptococcus      17
    SARS               16
    Pneumocystis       15
    COVID-19, ARDS     12
    ARDS                4
    E.Coli              4
    No Finding          3
    Chlamydophila       2
    Legionella          2
    Klebsiella          1
    '''
    is_covid = 'COVID-19' in finding
    if is_covid:
        return 'COVID'
    else:
        return 'Other'

CONTEXT = "Detecting Pneumonia and COVID-19 from Chest X-Ray Images"
TARGET = CuratedTarget(raw_name=LABEL_NAME, task_type=SupervisedTask.BINARY)
COLS_TO_DROP = ["url", 'patientid', 'Unnamed  28', 'doi', 'other_notes', 'license', 'folder',
                # Leakages
                'clinical_notes',
                ]
FEATURES = [CuratedFeature(raw_name=IMAGE_FEATURE_NAME, feat_type=FeatureType.IMAGE),
            CuratedFeature(raw_name="clinical_notes", feat_type=FeatureType.TEXT)]
IMAGE_FOLDER = "images"
LOADING_FUNC = load_df
