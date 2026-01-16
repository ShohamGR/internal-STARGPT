from os.path import exists, join

from pandas import DataFrame, read_csv

from tabstar_paper.datasets.curation_objects import CuratedTarget, CuratedFeature
from tabstar_paper.datasets.kaggle_competitions import download_kaggle_competition, KAGGLE_CACHE_DIR
from tabstar_paper.datasets.objects import SupervisedTask, FeatureType
from tabstar_paper.utils.io_handlers import unzip_file

COMPETITION_NAME = "petfinder-adoption-prediction"
COMPETITION_FOLDER = join(KAGGLE_CACHE_DIR, COMPETITION_NAME)
LABEL_NAME = "AdoptionSpeed"
IMAGE_FEATURE_PREFIX = 'Pet Image No '
N_IMAGES = 10

'''
Dataset Name: c/petfinder-adoption-prediction/
====
Examples: 14993
====
URL: https://www.kaggle.com/c/petfinder-adoption-prediction
====
Millions of stray animals suffer on the streets or are euthanized in shelters every day around the world. If homes can be found for them, many precious lives can be saved — and more happy families created.

PetFinder.my has been Malaysia’s leading animal welfare platform since 2008, with a database of more than 150,000 animals. PetFinder collaborates closely with animal lovers, media, corporations, and global organizations to improve animal welfare.

Animal adoption rates are strongly correlated to the metadata associated with their online profiles, such as descriptive text and photo characteristics. As one example, PetFinder is currently experimenting with a simple AI tool called the Cuteness Meter, which ranks how cute a pet is based on qualities present in their photos.

In this competition you will be developing algorithms to predict the adoptability of pets - specifically, how quickly is a pet adopted? If successful, they will be adapted into AI tools that will guide shelters and rescuers around the world on improving their pet profiles' appeal, reducing animal suffering and euthanization.

Top participants may be invited to collaborate on implementing their solutions into AI tools for assessing and improving pet adoption performance, which will benefit global animal welfare.

Description: 

Dataset Description
In this competition you will predict the speed at which a pet is adopted, based on the pet’s listing on PetFinder. Sometimes a profile represents a group of pets. In this case, the speed of adoption is determined by the speed at which all of the pets are adopted. The data included text, tabular, and image data. See below for details.
This is a Kernels-only competition. At the end of the competition, test data will be replaced in their entirety with new data of approximately the same size, and your kernels will be rerun on the new data.

File descriptions
train.csv - Tabular/text data for the training set
test.csv - Tabular/text data for the test set
sample_submission.csv - A sample submission file in the correct format
breed_labels.csv - Contains Type, and BreedName for each BreedID. Type 1 is dog, 2 is cat.
color_labels.csv - Contains ColorName for each ColorID
state_labels.csv - Contains StateName for each StateID
Data Fields
PetID - Unique hash ID of pet profile
AdoptionSpeed - Categorical speed of adoption. Lower is faster. This is the value to predict. See below section for more info.
Type - Type of animal (1 = Dog, 2 = Cat)
Name - Name of pet (Empty if not named)
Age - Age of pet when listed, in months
Breed1 - Primary breed of pet (Refer to BreedLabels dictionary)
Breed2 - Secondary breed of pet, if pet is of mixed breed (Refer to BreedLabels dictionary)
Gender - Gender of pet (1 = Male, 2 = Female, 3 = Mixed, if profile represents group of pets)
Color1 - Color 1 of pet (Refer to ColorLabels dictionary)
Color2 - Color 2 of pet (Refer to ColorLabels dictionary)
Color3 - Color 3 of pet (Refer to ColorLabels dictionary)
MaturitySize - Size at maturity (1 = Small, 2 = Medium, 3 = Large, 4 = Extra Large, 0 = Not Specified)
FurLength - Fur length (1 = Short, 2 = Medium, 3 = Long, 0 = Not Specified)
Vaccinated - Pet has been vaccinated (1 = Yes, 2 = No, 3 = Not Sure)
Dewormed - Pet has been dewormed (1 = Yes, 2 = No, 3 = Not Sure)
Sterilized - Pet has been spayed / neutered (1 = Yes, 2 = No, 3 = Not Sure)
Health - Health Condition (1 = Healthy, 2 = Minor Injury, 3 = Serious Injury, 0 = Not Specified)
Quantity - Number of pets represented in profile
Fee - Adoption fee (0 = Free)
State - State location in Malaysia (Refer to StateLabels dictionary)
RescuerID - Unique hash ID of rescuer
VideoAmt - Total uploaded videos for this pet
PhotoAmt - Total uploaded photos for this pet
Description - Profile write-up for this pet. The primary language used is English, with some in Malay or Chinese.
AdoptionSpeed
Contestants are required to predict this value. The value is determined by how quickly, if at all, a pet is adopted. The values are determined in the following way:
0 - Pet was adopted on the same day as it was listed.
1 - Pet was adopted between 1 and 7 days (1st week) after being listed.
2 - Pet was adopted between 8 and 30 days (1st month) after being listed.
3 - Pet was adopted between 31 and 90 days (2nd & 3rd month) after being listed.
4 - No adoption after 100 days of being listed. (There are no pets in this dataset that waited between 90 and 100 days).

Images
For pets that have photos, they will be named in the format of PetID-ImageNumber.jpg. Image 1 is the profile (default) photo set for the pet. For privacy purposes, faces, phone numbers and emails have been masked.

Image Metadata
We have run the images through Google's Vision API, providing analysis on Face Annotation, Label Annotation, Text Annotation and Image Properties. You may optionally utilize this supplementary information for your image analysis.

File name format is PetID-ImageNumber.json.

Some properties will not exist in JSON file if not present, i.e. Face Annotation. Text Annotation has been simplified to just 1 entry of the entire text description (instead of the detailed JSON result broken down by individual characters and words). Phone numbers and emails are already anonymized in Text Annotation.

Google Vision API reference: https://cloud.google.com/vision/docs/reference/rest/v1/images/annotate

Sentiment Data
We have run each pet profile's description through Google's Natural Language API, providing analysis on sentiment and key entities. You may optionally utilize this supplementary information for your pet description analysis. There are some descriptions that the API could not analyze. As such, there are fewer sentiment files than there are rows in the dataset.

File name format is PetID.json.

Google Natural Language API reference: https://cloud.google.com/natural-language/docs/basics

What will change in the 2nd stage of the competition?
In the second stage of the competition, we will re-run your selected Kernels. The following files will be swapped with new data:

test.zip including test.csv and sample_submission.csv
test_images.zip
test_metadata.zip
test_sentiment.zip
In stage 2, all data will be replaced with approximately the same amount of different data. The stage 1 test data will not be available when kernels are rerun in stage 2.

====
Target Variable: AdoptionSpeed (object, 5 distinct): ['Not adopted in 100 days', '8-30 Days', '31-90 Days', '1-7 Days', 'Same Day']
====
Features:

Type (int64, 2 distinct): ['1', '2']
Name (object, 9059 distinct): ['Baby', 'Lucky', 'No Name', 'Brownie', 'Mimi', 'Blackie', 'Puppy', 'Max', 'Kittens', 'Kitty']
Age (int64, 106 distinct): ['2', '1', '3', '4', '12', '24', '5', '6', '36', '8']
Breed1 (object, 176 distinct): ['Mixed Breed', 'Domestic Short Hair', 'Domestic Medium Hair', 'Tabby', 'Domestic Long Hair', 'Siamese', 'Persian', 'Labrador Retriever', 'Shih Tzu', 'Poodle']
Breed2 (object, 135 distinct): ['', 'Mixed Breed', 'Domestic Short Hair', 'Domestic Medium Hair', 'Tabby', 'Domestic Long Hair', 'Siamese', 'Terrier', 'Labrador Retriever', 'Persian']
Gender (int64, 3 distinct): ['2', '1', '3']
Color1 (object, 7 distinct): ['Black', 'Brown', 'Golden', 'Cream', 'Gray', 'White', 'Yellow']
Color2 (object, 7 distinct): ['', 'White', 'Brown', 'Cream', 'Gray', 'Yellow', 'Golden']
Color3 (object, 6 distinct): ['', 'White', 'Cream', 'Gray', 'Yellow', 'Golden']
MaturitySize (int64, 4 distinct): ['2', '1', '3', '4']
FurLength (int64, 3 distinct): ['1', '2', '3']
Vaccinated (int64, 3 distinct): ['2', '1', '3']
Dewormed (int64, 3 distinct): ['1', '2', '3']
Sterilized (int64, 3 distinct): ['2', '1', '3']
Health (int64, 3 distinct): ['1', '2', '3']
Quantity (int64, 19 distinct): ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
Fee (int64, 74 distinct): ['0', '50', '100', '200', '150', '20', '300', '30', '250', '1']
State (object, 14 distinct): ['Selangor', 'Kuala Lumpur', 'Pulau Pinang', 'Johor', 'Perak', 'Negeri Sembilan', 'Melaka', 'Kedah', 'Pahang', 'Terengganu']
VideoAmt (int64, 9 distinct): ['0', '1', '2', '3', '4', '5', '6', '8', '7']
Description (object, 14031 distinct): ['For Adoption', 'Dog 4 Adoption', 'Cat for adoption', 'Friendly', 'Dog for adoption', 'Please feel free to contact us : Stuart', 'PLEASE RESCUE/ADOPT ME FROM KLANG POUND OR I WILL BE PUT TO DEATH BY THIS WEEK, 28/3/10. I don\'t want to die,and I will love you immensely for saving me. Help!!! Please call ----------------------------------------------------- Adoption Procedure: This dog has been caught by Majlis Perbandaran Klang, and if nobody comes forward to adopt it, it will be euthanized within a few days. Even owned dogs are also often caught, and the owners are not aware for it. Those wishing to adopt this pet from Klang Dog Pound, please follow the procedures below: 1. Drive to Pusat Kurungan Haiwan Lebuh Sultan Muhammad Kawasan Perindustrian Bandar Sultan Sulaiman Pelabuhan Klang Tel : (For Sat & Sun, opening hours are 8am - 12pm) 2. Secure a Borang Permohonan Tuntutan Anjing, Selepas Tempoh 7 hari. Complete it & ensure it is endorsed by the relevant officier & stamped with relevant chop. 3. Provide a photostated copy of your Identification Card or Passport with each application * policies & requirements stiffen day by day * Advisable to provide a copy of IC/Passport per application (Just in case) * Secure extra application if there is any inkling of additional adoption. * Don\'t expect any leniency (Even we committee members, slaves & beggars don\'t have any unless OK by big guy) 4. Please be compassionate. Put yourself in their shoes: locked inside knowing its over. THEY DO KNOW. 5. I have seen them wasted much close to D days. Don\'t tell me they didn\'t undergo heightened enxiety & despair in anticipation of the end. What\'s worse their owners never came for them. Directions to Klang Dog Pound ================================ 1) Use Kesas Highway 2) Head for North Port till you see the signboard that writes "Melbourne 14 Days", then turn Right 3) Keep Left and turn Left at traffic light 4) Stay beside flyover and turn Right at immediate traffic light 5) Drive towards Sultan Sulaiman Industrial Estate 6) Go up first set of flyover 7) Keep Left till you see Pusat Kurungan Haiwan signboard 8) Turn Left 9) Drive on till you see gravel road work beside retention pond at the right 10) Turn in and turn Right till you reach a blue-roofed pound', 'I need a new home!! Contact Furry Friends Farm if you want to adopt me.', "The lil' puppy is currently taking shelter at SPCA Seberang Perai. Those interested to adopt her may contact us via email.", 'The puppy is currently taking shelter at SPCA Seberang Perai. Please contact SPCA Seberang Perai if you are interested to adopt her as your pet.']
PhotoAmt (float64, 31 distinct): ['1.0', '2.0', '3.0', '5.0', '4.0', '6.0', '7.0', '0.0', '8.0', '9.0']
Pet Image No 1 (object, 14653 distinct): ['', '0dff88b6f-1.jpg', '6f18704f7-1.jpg', '093cce3a1-1.jpg', 'aa91c3400-1.jpg', 'e86af84c5-1.jpg', '77a6f71e8-1.jpg', '2885ade43-1.jpg', '5d3289417-1.jpg', '75b726ba0-1.jpg']
Pet Image No 2 (object, 11578 distinct): ['', '6296e909a-2.jpg', '3422e4906-2.jpg', '5842f1ff5-2.jpg', '850a43f90-2.jpg', 'd24c30b4b-2.jpg', '1caa6fcdb-2.jpg', '97aa9eeac-2.jpg', 'c06d167ca-2.jpg', '7a0942d61-2.jpg']
Pet Image No 3 (object, 9060 distinct): ['', '3422e4906-3.jpg', '5842f1ff5-3.jpg', '850a43f90-3.jpg', '1caa6fcdb-3.jpg', '97aa9eeac-3.jpg', 'c06d167ca-3.jpg', '8b693ca84-3.jpg', '1c92ce464-3.jpg', '6436c1a59-3.jpg']
Pet Image No 4 (object, 6549 distinct): ['', '3422e4906-4.jpg', '5842f1ff5-4.jpg', '97aa9eeac-4.jpg', 'c06d167ca-4.jpg', '8b693ca84-4.jpg', '1c92ce464-4.jpg', '6436c1a59-4.jpg', '234a5a54c-4.jpg', '988988d5b-4.jpg']
Pet Image No 5 (object, 4668 distinct): ['', '3422e4906-5.jpg', '5842f1ff5-5.jpg', '97aa9eeac-5.jpg', 'c06d167ca-5.jpg', '8b693ca84-5.jpg', '1c92ce464-5.jpg', '6436c1a59-5.jpg', '234a5a54c-5.jpg', '988988d5b-5.jpg']
Pet Image No 6 (object, 2521 distinct): ['', '3422e4906-6.jpg', '5842f1ff5-6.jpg', '97aa9eeac-6.jpg', 'c06d167ca-6.jpg', '8b693ca84-6.jpg', '1c92ce464-6.jpg', '6436c1a59-6.jpg', '988988d5b-6.jpg', '85fc3c314-6.jpg']
Pet Image No 7 (object, 1900 distinct): ['', '3422e4906-7.jpg', '5842f1ff5-7.jpg', '97aa9eeac-7.jpg', '8b693ca84-7.jpg', '1c92ce464-7.jpg', '6436c1a59-7.jpg', '988988d5b-7.jpg', '85fc3c314-7.jpg', '9415bc79e-7.jpg']
Pet Image No 8 (object, 1468 distinct): ['', '5842f1ff5-8.jpg', '97aa9eeac-8.jpg', '1c92ce464-8.jpg', '988988d5b-8.jpg', '85fc3c314-8.jpg', 'ace21b6b1-8.jpg', 'c1e816568-8.jpg', '22f6c0ac6-8.jpg', '6fecc5cf7-8.jpg']
Pet Image No 9 (object, 1154 distinct): ['', '97aa9eeac-9.jpg', '988988d5b-9.jpg', '85fc3c314-9.jpg', 'ace21b6b1-9.jpg', 'c1e816568-9.jpg', '22f6c0ac6-9.jpg', '6fecc5cf7-9.jpg', 'f6551137d-9.jpg', 'b0dec8779-9.jpg']
Pet Image No 10 (object, 923 distinct): ['', '988988d5b-10.jpg', '85fc3c314-10.jpg', 'ace21b6b1-10.jpg', 'c1e816568-10.jpg', '22f6c0ac6-10.jpg', '6fecc5cf7-10.jpg', 'f6551137d-10.jpg', 'b0dec8779-10.jpg', 'fd3172797-10.jpg']
'''


def load_df(dir_path: str) -> DataFrame:
    if not exists(COMPETITION_FOLDER):
        download_kaggle_competition(competition=COMPETITION_NAME)
        zip_file = f"{COMPETITION_FOLDER}.zip"
        unzip_file(zip_file)
    df = _get_csv("train/train.csv")
    breed = _get_csv("breed_labels.csv")
    breed = breed.set_index('BreedID')['BreedName'].to_dict()
    color = _get_csv("color_labels.csv")
    color = color.set_index('ColorID')['ColorName'].to_dict()
    state = _get_csv("state_labels.csv")
    state = state.set_index('StateID')['StateName'].to_dict()
    for col_name, col_dict in [('Breed1', breed), ('Breed2', breed), ('Color1', color),
                               ('Color2', color), ('Color3', color), ('State', state)]:
        df = from_dict(df, col_name, mapping=col_dict)
    # Every pet might have more than N_IMAGES pictures, but we will limit ourselves to the top ones
    for n in range(1, N_IMAGES + 1):
        df[f'{IMAGE_FEATURE_PREFIX}{n}'] = df['PetID'].apply(lambda x: path_if_exists(x=x, n=n))
    df[LABEL_NAME] = df[LABEL_NAME].apply(map_label)
    df.drop(columns=['RescuerID', 'PetID'], inplace=True)
    return df


def _get_csv(csv: str) -> DataFrame:
    return read_csv(join(COMPETITION_FOLDER, csv))

def from_dict(df: DataFrame, col: str, mapping: dict) -> DataFrame:
    df[col] = df[col].apply(lambda x: mapping.get(x, ''))
    return df

def path_if_exists(x: str, n: int) -> str:
    img_name = f"{x}-{n}.jpg"
    path = join(IMAGE_FOLDER, img_name)
    if exists(path):
        return img_name
    return ""


def map_label(i: int) -> str:
    mapping = {0: 'Same Day', 1: '1-7 Days', 2: '8-30 Days', 3: '31-90 Days', 4: 'Not adopted in 100 days'}
    return mapping[i]

CONTEXT = "PetFinder prediction speed of adoption of pets based on images, text and tabular data."
TARGET = CuratedTarget(raw_name=LABEL_NAME, task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = []
TEXT_FEATURES = [CuratedFeature(raw_name=f, feat_type=FeatureType.TEXT) for f in ['Name', 'Description', 'Breed1', 'Breed2']]
FEATURES = [CuratedFeature(raw_name=f"{IMAGE_FEATURE_PREFIX}{n}", feat_type=FeatureType.IMAGE) for n in range(1, N_IMAGES + 1)] + TEXT_FEATURES
IMAGE_FOLDER = join(COMPETITION_FOLDER, "train_images")
LOADING_FUNC = load_df
