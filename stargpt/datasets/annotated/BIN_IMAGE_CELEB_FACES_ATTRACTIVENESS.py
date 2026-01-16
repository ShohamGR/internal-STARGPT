from pandas import DataFrame

from tabstar2.utils.datasets import load_csv
from tabstar_paper.datasets.curation_objects import CuratedTarget, CuratedFeature
from tabstar_paper.datasets.objects import SupervisedTask, FeatureType

'''
Dataset Name: jessicali9530/celeba-dataset/
====
Examples: 202599
====
URL: https://www.kaggle.com/jessicali9530/celeba-dataset
====
From The CHARMS Paper:
CelebA. CelebA is the abbreviation of CelebFaces Attribute, meaning celebrity face attribute dataset, which contains
202, 599 face images of 10, 177 celebrities, each image is well marked with features, including 40 attribute markers such
as Big Nose. We use Attractive as the label, which is a binary classification task. We use 8 : 1 : 1 to divide the training
set, validation set, and testing set. There are total of 39 categorical variables in this dataset. We expect to introduce more
detailed face information in the table, allowing the image to perform better on downstream tasks.

Description: 
About Dataset
Context
A popular component of computer vision and deep learning revolves around identifying faces for various applications from logging into your phone with your face or searching through surveillance images for a particular suspect. This dataset is great for training and testing models for face detection, particularly for recognising facial attributes such as finding people with brown hair, are smiling, or wearing glasses. Images cover large pose variations, background clutter, diverse people, supported by a large quantity of images and rich annotations. This data was originally collected by researchers at MMLAB, The Chinese University of Hong Kong (specific reference in Acknowledgment section).

Content
Overall

202,599 number of face images of various celebrities
10,177 unique identities, but names of identities are not given
40 binary attribute annotations per image
5 landmark locations
Data Files

img_align_celeba.zip: All the face images, cropped and aligned
list_eval_partition.csv: Recommended partitioning of images into training, validation, testing sets. Images 1-162770 are training, 162771-182637 are validation, 182638-202599 are testing
list_bbox_celeba.csv: Bounding box information for each image. "x_1" and "y_1" represent the upper left point coordinate of bounding box. "width" and "height" represent the width and height of bounding box
list_landmarks_align_celeba.csv: Image landmarks and their respective coordinates. There are 5 landmarks: left eye, right eye, nose, left mouth, right mouth
list_attr_celeba.csv: Attribute labels for each image. There are 40 attributes. "1" represents positive while "-1" represents negative
Acknowledgements
Original data and banner image source came from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
As mentioned on the website, the CelebA dataset is available for non-commercial research purposes only. For specifics please refer to the website.

The creators of this dataset wrote the following paper employing CelebA for face detection:

S. Yang, P. Luo, C. C. Loy, and X. Tang, "From Facial Parts Responses to Face Detection: A Deep Learning Approach", in IEEE International Conference on Computer Vision (ICCV), 2015

Inspiration
Can you train a model that can detect particular facial attributes?
Which images contain people that are smiling?
Does someone have straight or wavy hair?

====
Target Variable: Attractive (int64, 2 distinct): ['1', '-1']
====
Features:

image_id (object, 202599 distinct): ['000001.jpg', '000002.jpg', '000003.jpg', '000004.jpg', '000005.jpg', '000006.jpg', '000007.jpg', '000008.jpg', '000009.jpg', '000010.jpg']
5_o_Clock_Shadow (int64, 2 distinct): ['-1', '1']
Arched_Eyebrows (int64, 2 distinct): ['-1', '1']
Bags_Under_Eyes (int64, 2 distinct): ['-1', '1']
Bald (int64, 2 distinct): ['-1', '1']
Bangs (int64, 2 distinct): ['-1', '1']
Big_Lips (int64, 2 distinct): ['-1', '1']
Big_Nose (int64, 2 distinct): ['-1', '1']
Black_Hair (int64, 2 distinct): ['-1', '1']
Blond_Hair (int64, 2 distinct): ['-1', '1']
Blurry (int64, 2 distinct): ['-1', '1']
Brown_Hair (int64, 2 distinct): ['-1', '1']
Bushy_Eyebrows (int64, 2 distinct): ['-1', '1']
Chubby (int64, 2 distinct): ['-1', '1']
Double_Chin (int64, 2 distinct): ['-1', '1']
Eyeglasses (int64, 2 distinct): ['-1', '1']
Goatee (int64, 2 distinct): ['-1', '1']
Gray_Hair (int64, 2 distinct): ['-1', '1']
Heavy_Makeup (int64, 2 distinct): ['-1', '1']
High_Cheekbones (int64, 2 distinct): ['-1', '1']
Male (int64, 2 distinct): ['-1', '1']
Mouth_Slightly_Open (int64, 2 distinct): ['-1', '1']
Mustache (int64, 2 distinct): ['-1', '1']
Narrow_Eyes (int64, 2 distinct): ['-1', '1']
No_Beard (int64, 2 distinct): ['1', '-1']
Oval_Face (int64, 2 distinct): ['-1', '1']
Pale_Skin (int64, 2 distinct): ['-1', '1']
Pointy_Nose (int64, 2 distinct): ['-1', '1']
Receding_Hairline (int64, 2 distinct): ['-1', '1']
Rosy_Cheeks (int64, 2 distinct): ['-1', '1']
Sideburns (int64, 2 distinct): ['-1', '1']
Smiling (int64, 2 distinct): ['-1', '1']
Straight_Hair (int64, 2 distinct): ['-1', '1']
Wavy_Hair (int64, 2 distinct): ['-1', '1']
Wearing_Earrings (int64, 2 distinct): ['-1', '1']
Wearing_Hat (int64, 2 distinct): ['-1', '1']
Wearing_Lipstick (int64, 2 distinct): ['-1', '1']
Wearing_Necklace (int64, 2 distinct): ['-1', '1']
Wearing_Necktie (int64, 2 distinct): ['-1', '1']
Young (int64, 2 distinct): ['1', '-1']
'''

def load_df(dir_path: str) -> DataFrame:
    df = load_csv(dir_path, "list_attr_celeba.csv")
    return df


CONTEXT = ""
TARGET = CuratedTarget(raw_name='Attractive', task_type=SupervisedTask.BINARY)
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name='image_id', feat_type=FeatureType.IMAGE)]
IMAGE_FOLDER = "img_align_celeba/img_align_celeba"
LOADING_FUNC = load_df
