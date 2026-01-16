import os
from os.path import join

from pandas import concat, DataFrame

from tabstar2.utils.datasets import load_csv
from tabstar_paper.datasets.curation_objects import CuratedTarget, CuratedFeature
from tabstar_paper.datasets.objects import SupervisedTask, FeatureType

'''
Dataset Name: shreyanmohanty/oasis-alzheimers-detection-multi-class-dataset/
====
Examples: 488
====
URL: https://www.kaggle.com/shreyanmohanty/oasis-alzheimers-detection-multi-class-dataset
====
Description: 
About Dataset
Based on the OASIS-1 Alzheimer's Dataset with 416 subjects.

Most public Alzheimer's datasets exhibit training data leakage, since MRI slices from the same patient are randomly allocated in the test/train split. Due to class imbalance, it become difficult to evenly split the dataset with scikit-learn libraries. This dataset randomly choses patients from each class and allocates them into the training set at an amount deemed sufficient to learn features from. Due to the massive imbalance caused by the "NonDemented" class, which has more images than some other classes combines, we put the majority of these patients in the test set. It is not necessary to have so many examples for successful training.

Acknowledgements: "Yiwei Lu has performed image conversion along with skull striping and other tissue removal with their pre-trained LinkNet3D model."

Citation: Well-Documented Alzheimer's Dataset: https://doi.org/10.34740/kaggle/dsv/10215637

I have cropped and resized the images to 224 x 224 (with padding), making it very easy to just plug this dataset in and get started fine-tuning with pre-trained models. Roboflow is used to augment data, the specifics are provided in a txt file in both the test and train directories. Please read the provenance section for detailed information. There is still a class imbalance, which will probably require weighted sampling or weighted loss during the training process.

Mean and Standard Deviation values for Normalization (Obtained using training set only):

Mean: [0.2682, 0.2682, 0.2682]

Std: [0.3008, 0.3008, 0.3008]

Acknowledgments: “Data were provided 1-12 by OASIS-1: Cross-Sectional: Principal Investigators: D. Marcus, R, Buckner, J, Csernansky J. Morris; P50 AG05681, P01 AG03991, P01 AG026276, R01 AG021910, P20 MH071616, U24 RR021382”

Citation: OASIS-1: Cross-Sectional: https://doi.org/10.1162/jocn.2007.19.9.1498
====
Target Variable: is_demented (object, 2 distinct): ['NonDemented', 'Demented']
====
Features:

ID (object, 244 distinct): ['test/NonDemented/OAS1_0001_MR1_3_nii_slice_115_png.rf.ce73e16d846e609aff4a616ebda9ae95.jpg', 'test/VeryMildDemented/OAS1_0003_MR1_1_nii_slice_175_png.rf.fd57c114391c2c4bd14067d775b000f5.jpg', 'test/NonDemented/OAS1_0004_MR1_1_nii_slice_110_png.rf.6b78fb8657a7ffb40201462168b2a54b.jpg', 'test/NonDemented/OAS1_0005_MR1_1_nii_slice_99_png.rf.7d19928aa5d3988a0c03e34f1b54f40e.jpg', 'test/NonDemented/OAS1_0006_MR1_4_nii_slice_119_png.rf.981e45abb5c7852c85f23c588da51f41.jpg', 'test/NonDemented/OAS1_0011_MR1_4_nii_slice_156_png.rf.97910e8bcc08bce8b6118d02837ff05d.jpg', 'test/VeryMildDemented/OAS1_0016_MR1_3_nii_slice_113_png.rf.7a83824caf25ac4806e4206517decff0.jpg', 'test/NonDemented/OAS1_0017_MR1_2_nii_slice_131_png.rf.0ff6272afb2182429a1e6deab9b94fdf.jpg', 'test/NonDemented/OAS1_0018_MR1_4_nii_slice_154_png.rf.812c4ed33eb57cea1f05b75fb61cbaf5.jpg', 'test/NonDemented/OAS1_0019_MR1_1_nii_slice_110_png.rf.bb7ae2ac9fd6cd59e6d34ba060bbcac8.jpg']
M/F (object, 2 distinct): ['F', 'M']
Age (int64, 67 distinct): ['22', '20', '23', '80', '21', '73', '48', '19', '24', '26']
Educ (float64, 5 distinct, 50.4% missing): ['2.0', '4.0', '3.0', '5.0', '1.0']
SES (float64, 5 distinct, 52.5% missing): ['3.0', '2.0', '1.0', '4.0', '5.0']
eTIV (int64, 199 distinct): ['1346', '1536', '1501', '1714', '1295', '1475', '1561', '1444', '1373', '1582']
ASF (float64, 186 distinct): ['1.142', '1.169', '1.19', '1.216', '1.181', '1.17', '1.124', '1.024', '1.304', '0.978']
Delay (float64, 10 distinct, 95.9% missing): ['1.0', '5.0', '64.0', '2.0', '10.0', '12.0', '21.0', '24.0', '3.0', '39.0']
'''

IS_DEMENTED = "is_demented"
ID = "ID"


def load_df(dir_path: str) -> DataFrame:
    train_df = load_csv(dir_path, "oasis_test_patients_metadata.csv")
    test_df = load_csv(dir_path, "oasis_test_patients_metadata.csv")
    df = concat([train_df, test_df])
    df[IS_DEMENTED] = df['class'].apply(lambda x: x if x == 'NonDemented' else 'Demented')
    df.drop(columns=['class'], inplace=True, errors='ignore')
    collect_image_paths(df, dir_path=dir_path)
    return df


def collect_image_paths(df: DataFrame, dir_path: str) -> None:
    # Files look like this: 'OAS1_0247_MR1_1-nii_slice_129_png.rf.fab22c473e5c75a743456ea216849dac.jpg'
    prefix2image = {}
    for split in ['train', 'test']:
        for condition in ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']:
            condition_path = join(split, condition)
            for f in os.listdir(join(dir_path, condition_path)):
                img_path = join(condition_path, f)
                file_prefix = f[:13]
                prefix2image.setdefault(file_prefix, img_path)
    df[ID] = df[ID].map(prefix2image)


CONTEXT = ""
TARGET = CuratedTarget(raw_name=IS_DEMENTED, task_type=SupervisedTask.BINARY)
COLS_TO_DROP = [
    # Constant
    'Hand',
    # Leakage
    "CDR", "MMSE", "nWBV"]
FEATURES = [CuratedFeature(raw_name=ID, feat_type=FeatureType.IMAGE)]
IMAGE_FOLDER = ""
LOADING_FUNC = load_df
