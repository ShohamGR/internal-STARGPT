import os
from os.path import join
from typing import List

import pandas as pd
from pandas import DataFrame

from tabstar2.utils.datasets import load_csv
from tabstar_paper.datasets.curation_objects import CuratedTarget, CuratedFeature
from tabstar_paper.datasets.objects import SupervisedTask, FeatureType

'''
Dataset Name: awsaf49/cbis-ddsm-breast-cancer-image-dataset/
====
Examples: 1696
====
URL: https://www.kaggle.com/awsaf49/cbis-ddsm-breast-cancer-image-dataset
====
Description: 
CBIS-DDSM: Breast Cancer Image Dataset
Curated Breast Imaging Subset DDSM Dataset (Mammography)

About Dataset


Descripton
This dataset is jpeg format of the original dataset(163GB). The resolution was kept to the original dataset.

Number of Studies: 6775
Number of Series: 6775
Number of Participants: 1,566(NB)
Number of Images: 10239
Modalities: MG
Image Size (GB): 6(.jpg)
NB: The image data for this collection is structured such that each participant has multiple patient IDs. For example, pat_id 00038 has 10 separate patient IDs which provide information about the scans within the IDs (e.g. Calc-Test_P_00038_LEFT_CC, Calc-Test_P_00038_RIGHT_CC_1) This makes it appear as though there are 6,671 participants according to the DICOM metadata, but there are only 1,566 actual participants in the cohort.

Summary
This CBIS-DDSM (Curated Breast Imaging Subset of DDSM) is an updated and standardized version of the Digital Database for Screening Mammography (DDSM). The DDSM is a database of 2,620 scanned film mammography studies. It contains normal, benign, and malignant cases with verified pathology information. The scale of the database along with ground truth validation makes the DDSM a useful tool in the development and testing of decision support systems. The CBIS-DDSM collection includes a subset of the DDDSM data selected and curated by a trained mammographer. The images have been decompressed and converted to DICOM format. Updated ROI segmentation and bounding boxes, and pathologic diagnosis for training data are also included. A manuscript describing how to use this dataset in detail is available at https://www.nature.com/articles/sdata2017177.

Published research results from work in developing decision support systems in mammography are difficult to replicate due to the lack of a standard evaluation data set; most computer-aided diagnosis (CADx) and detection (CADe) algorithms for breast cancer in mammography are evaluated on private data sets or on unspecified subsets of public databases. Few well-curated public datasets have been provided for the mammography community. These include the DDSM, the Mammographic Imaging Analysis Society (MIAS) database, and the Image Retrieval in Medical Applications (IRMA) project. Although these public data sets are useful, they are limited in terms of data set size and accessibility.

For example, most researchers using the DDSM do not leverage all its images for a variety of historical reasons. When the database was released in 1997, computational resources to process hundreds or thousands of images were not widely available. Additionally, the DDSM images are saved in non-standard compression files that require the use of decompression code that has not been updated or maintained for modern computers. Finally, the ROI annotations for the abnormalities in the DDSM were provided to indicate a general position of lesions, but not a precise segmentation for them. Therefore, many researchers must implement segmentation algorithms for accurate feature extraction. This causes an inability to directly compare the performance of methods or to replicate prior results. The CBIS-DDSM collection addresses that challenge by publicly releasing a curated and standardized version of the DDSM for evaluation of future CADx and CADe systems (sometimes referred to generally as CAD) research in mammography.

Please note that the image data for this collection is structured such that each participant has multiple patient IDs. For example, participant 00038 has 10 separate patient IDs which provide information about the scans within the IDs (e.g. Calc-Test_P_00038_LEFT_CC, Calc-Test_P_00038_RIGHT_CC_1). This makes it appear as though there are 6,671 patients according to the DICOM metadata, but there are only 1,566 actual participants in the cohort.

For scientific inquiries about this dataset, please contact Dr. Daniel Rubin, Department of Biomedical Data Science, Radiology, and Medicine, Stanford University School of Medicine (dlrubin@stanford.edu).

Citations & Data Usage Policy
Users of this data must abide by the TCIA Data Usage Policy and the Creative Commons Attribution 3.0 Unported License under which it has been published. Attribution should include references to the following citations:

CBIS-DDSM Citation
 Rebecca Sawyer Lee, Francisco Gimenez, Assaf Hoogi , Daniel Rubin  (2016). **Curated Breast Imaging Subset of DDSM [Dataset]**. The Cancer Imaging Archive. **DOI:**  https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY
Publication Citation
Rebecca Sawyer Lee, Francisco Gimenez, Assaf Hoogi, Kanae Kawai Miyake, Mia Gorovoy & Daniel L. Rubin. (2017) **A curated mammography data set for use in computer-aided detection and diagnosis research**. Scientific Data volume 4, Article number: 170177 DOI: https://doi.org/10.1038/sdata.2017.177
content_copy
TCIA Citation
Clark K, Vendt B, Smith K, Freymann J, Kirby J, Koppel P, Moore S, Phillips S, Maffitt D, Pringle M, Tarbox L, Prior F. The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository, Journal of Digital Imaging, Volume 26, Number 6, December, 2013, pp 1045-1057. DOI: https://doi.org/10.1007/s10278-013-9622-7*

From MultiModalPFN Paper:

CBIS–DDSM The CBIS–DDSM(Sawyer-Lee et al., 2016) dataset is a curated subset of the
Digital Database for Screening Mammography (DDSM), designed to support computer-aided
detection and diagnosis of breast cancer. It consists of digitized film mammography images
with annotated regions of interest for two primary lesion types: calcifications and masses.
Each case is associated with pathologic diagnosis labels (benign or malignant) and includes detailed metadata such as lesion type, subtlety, and assessment category, with all malignant cases
biopsy-proven. The original target labels in CBIS–DDSM consist of three categories: MALIGNANT, BENIGN, and BENIGN WITHOUT CALLBACK. For this study, we merged BENIGN WITHOUT CALLBACK and BENIGN into a single class, thereby formulating a binary
classification task. Since the labels are well balanced, we report accuracy rather than ROC AUC for
binary classification. Prior work on CBIS–DDSM typically reports high accuracy by augmenting the dataset with additional external sources. In contrast, studies that rely solely on CBIS–
DDSM often employ strategies such as resampling the training and test splits to balance class
distributions before performing classification. In our experiments, however, we used the dataset
in its original form without any modifications.https://www.cancerimagingarchive.
net/collection/cbis-ddsm/, https://www.kaggle.com/datasets/awsaf49/
cbis-ddsm-breast-cancer-image-dataset (The Kaggle dataset contains the same images, but stored in resized JPEG format with smaller dimensions to reduce the overall dataset size.)

====
Target Variable: pathology (object, 3 distinct): ['MALIGNANT', 'BENIGN', 'BENIGN_WITHOUT_CALLBACK']
====
Features:

breast_density (int64, 4 distinct): ['2', '3', '1', '4']
left or right breast (object, 2 distinct): ['RIGHT', 'LEFT']
image view (object, 2 distinct): ['MLO', 'CC']
abnormality id (int64, 6 distinct): ['1', '2', '3', '4', '5', '6']
mass shape (object, 20 distinct, 0.2% missing): ['IRREGULAR', 'OVAL', 'LOBULATED', 'ROUND', 'ARCHITECTURAL_DISTORTION', 'IRREGULAR-ARCHITECTURAL_DISTORTION', 'LYMPH_NODE', 'ASYMMETRIC_BREAST_TISSUE', 'FOCAL_ASYMMETRIC_DENSITY', 'LOBULATED-IRREGULAR']
mass margins (object, 19 distinct, 3.5% missing): ['CIRCUMSCRIBED', 'ILL_DEFINED', 'SPICULATED', 'OBSCURED', 'MICROLOBULATED', 'ILL_DEFINED-SPICULATED', 'CIRCUMSCRIBED-ILL_DEFINED', 'OBSCURED-ILL_DEFINED', 'CIRCUMSCRIBED-OBSCURED', 'OBSCURED-ILL_DEFINED-SPICULATED']
assessment (int64, 6 distinct): ['4', '5', '3', '0', '2', '1']
subtlety (int64, 6 distinct): ['5', '4', '3', '2', '1', '0']
image file path (object, 1592 distinct): ['1.3.6.1.4.1.9590.100.1.2.87251504411596839017815563663575708222/1-219.jpg', '1.3.6.1.4.1.9590.100.1.2.354587724213018641829708719832963731890/1-179.jpg', '1.3.6.1.4.1.9590.100.1.2.383084015312187246035597241651391161847/1-124.jpg', '1.3.6.1.4.1.9590.100.1.2.260395375912689985505181352172038713429/1-061.jpg', '1.3.6.1.4.1.9590.100.1.2.292605978712963936606864280561587921668/1-031.jpg', '1.3.6.1.4.1.9590.100.1.2.101999469712679926627011488331183444331/1-063.jpg', '1.3.6.1.4.1.9590.100.1.2.170204618311705960537961724650736097259/1-020.jpg', '1.3.6.1.4.1.9590.100.1.2.252718871411822036139213438773416034416/1-212.jpg', '1.3.6.1.4.1.9590.100.1.2.220564602411627135838034738500414889195/1-044.jpg', '1.3.6.1.4.1.9590.100.1.2.154877164112814387839246109501067006440/1-231.jpg']
cropped image file path (object, 1696 distinct): ['1.3.6.1.4.1.9590.100.1.2.30820586311062570442302321942433426184/1-083.jpg', '1.3.6.1.4.1.9590.100.1.2.381440141511137044327302306604206077287/1-084.jpg', '1.3.6.1.4.1.9590.100.1.2.212143028513012144941507232513982203672/1-274.jpg', '1.3.6.1.4.1.9590.100.1.2.15403043813402510742192372832381918984/1-275.jpg', '1.3.6.1.4.1.9590.100.1.2.199593071810497070809647901570077988031/1-276.jpg', '1.3.6.1.4.1.9590.100.1.2.44610919611642954332266410812181604922/1-277.jpg', '1.3.6.1.4.1.9590.100.1.2.335564193512609498716387099372607181452/1-278.jpg', '1.3.6.1.4.1.9590.100.1.2.371211525812372459428004652923830584055/1-279.jpg', '1.3.6.1.4.1.9590.100.1.2.94222899611000951402700588920137869763/1-280.jpg', '1.3.6.1.4.1.9590.100.1.2.90316691111901811331698126462090336197/1-282.jpg']
'''

LABEL_NAME = ""
IMAGE_FEATURE_NAME = "image file path"
CROPPED_FEATURE = 'cropped image file path'


def load_df(dir_path: str) -> DataFrame:
    mass_case_train = load_csv(dir_path, "csv/mass_case_description_train_set.csv")
    mass_case_test = load_csv(dir_path, "csv/mass_case_description_test_set.csv")
    df = pd.concat([mass_case_test, mass_case_train])
    df[IMAGE_FEATURE_NAME] = df[IMAGE_FEATURE_NAME].apply(lambda im: extract_img_path(img_id=im, dir_path=dir_path))
    df[CROPPED_FEATURE] = df[CROPPED_FEATURE].apply(lambda im: extract_cropped(img_id=im, dir_path=dir_path))
    # Removing this unclear image, I think it's only the label for benign or something
    df.drop(columns=['ROI mask file path'], inplace=True)
    return df


def extract_img_path(img_id: str, dir_path: str) -> str:
    relevant_id = _extract_relevant_id(img_id)
    files = _get_files(relevant_id, dir_path)
    return _get_final_path(files, relevant_id)

def extract_cropped(img_id: str, dir_path: str) -> str:
    relevant_id = _extract_relevant_id(img_id)
    files = _get_files(relevant_id, dir_path)
    files = [f for f in files if f[0] == '1']
    return _get_final_path(files, relevant_id)

def _extract_relevant_id(img_id: str) -> str:
    assert img_id.count("/") == 3
    _, _, relevant_id, _ = img_id.split("/")
    return relevant_id

def _get_files(relevant_id: str, dir_path: str) -> List[str]:
    img_dir = join(dir_path, IMAGE_FOLDER)
    files = os.listdir(join(img_dir, relevant_id))
    return files

def _get_final_path(files: List[str], relevant_id: str) -> str:
    assert len(files) == 1
    pic_id = files[0]
    return join(relevant_id, pic_id)

CONTEXT = ""
TARGET = CuratedTarget(raw_name='pathology', task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = ["patient_id", "abnormality type"]
FEATURES = [CuratedFeature(raw_name=IMAGE_FEATURE_NAME, feat_type=FeatureType.IMAGE),
            CuratedFeature(raw_name=CROPPED_FEATURE, feat_type=FeatureType.IMAGE),]
IMAGE_FOLDER = "jpeg"
LOADING_FUNC = load_df
