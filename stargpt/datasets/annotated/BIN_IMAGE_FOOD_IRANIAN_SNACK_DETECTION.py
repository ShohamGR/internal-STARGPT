import os
from os.path import join
from typing import List

from pandas import DataFrame

from tabstar_paper.datasets.curation_objects import CuratedTarget, CuratedFeature
from tabstar_paper.datasets.objects import SupervisedTask, FeatureType
from tabstar_paper.utils.io_handlers import load_txt

LABEL_NAME = "Has Ashi Mashi"
IMAGE_FEATURE_NAME = "Snack"

'''
Dataset Name: halfbloodhamed/iranian-snack-and-chips-detection-yolo-format/
====
Examples: 603
====
URL: https://www.kaggle.com/halfbloodhamed/iranian-snack-and-chips-detection-yolo-format
====
Description: 
About Dataset
🧃 Iranian Snack & Chips Detection Dataset (YOLO Format)
This dataset contains annotated images of popular Iranian supermarket snacks and chips collected and labeled for object detection and instance segmentation tasks. It features 19 different product classes from well-known brands like Ashi Mashi, Cheetoz, Maz Maz, Naderi, Minoo, and Lina.

Perfect for training and evaluating YOLO models — including YOLOv5, YOLOv8, YOLOv11, and YOLOv12 — on real-world packaged goods in retail environments.
📁 Dataset Structure:

train/ – Training images
valid/ – Validation images
test/ – Test images
data.yaml – Configuration file for YOLO models

Annotations are in YOLO format (polygons)
🧠 Classes (19 Total):

['Ashi Mashi snacks', 'Chee pellet ketchup', 'Chee pellet vinegar',
'Cheetoz chili chips', 'Cheetoz ketchup chips', 'Cheetoz onion and parsley chips',
'Cheetoz salty chips', 'Cheetoz snack 30g', 'Cheetoz snack 90g',
'Cheetoz vinegar chips', 'Cheetoz wheelsnack', 'Maz Maz ketchup chips',
'Maz Maz potato sticks', 'Maz Maz salty chips', 'Maz Maz vinegar chips',

'Mini Lina', 'Minoo cream biscuit', 'Naderi mini cookie', 'Naderi mini wafer']
🔧 Recommended Use Cases:

Product recognition in retail and supermarket scenes
Fine-tuning YOLO models for regional or branded goods
Instance segmentation of snacks and chips

Dataset for research on real-world consumer product detection
📎 Source & License:

Annotated with Roboflow
License: CC BY 4.0 – Free to use, modify, and redistribute with attribution
Created by: Hamed Mahmoudi (halfbloodhamed)
====
Target Variable: Has Ashi Mashi (bool, 2 distinct): ['0', '1']
====
Features:

Snack (object, 603 distinct): ['/data/home/alan.arazi/.cache/kagglehub/datasets/halfbloodhamed/iranian-snack-and-chips-detection-yolo-format/versions/4/Iranian Snack and Chips Detection (YOLO Format)/train/images/20250506_211037_mp4-0544_jpg.rf.e34393140ee17c0180312c0eaf3ce7d6.jpg', '/data/home/alan.arazi/.cache/kagglehub/datasets/halfbloodhamed/iranian-snack-and-chips-detection-yolo-format/versions/4/Iranian Snack and Chips Detection (YOLO Format)/train/images/20250506_211037_mp4-0488_jpg.rf.c1b6116b8f5b7a2b42ea7c8579b587f6.jpg', '/data/home/alan.arazi/.cache/kagglehub/datasets/halfbloodhamed/iranian-snack-and-chips-detection-yolo-format/versions/4/Iranian Snack and Chips Detection (YOLO Format)/train/images/20250506_211037_mp4-0137_jpg.rf.f7f893e6f21a69b3b0750bcef372b056.jpg', '/data/home/alan.arazi/.cache/kagglehub/datasets/halfbloodhamed/iranian-snack-and-chips-detection-yolo-format/versions/4/Iranian Snack and Chips Detection (YOLO Format)/train/images/20250506_205726_mp4-0000_jpg.rf.40f4f9cd7d9337b595802251e489cc58.jpg', '/data/home/alan.arazi/.cache/kagglehub/datasets/halfbloodhamed/iranian-snack-and-chips-detection-yolo-format/versions/4/Iranian Snack and Chips Detection (YOLO Format)/train/images/20250506_211037_mp4-1055_jpg.rf.2f745efaacfb1f6955c500dff9848011.jpg', '/data/home/alan.arazi/.cache/kagglehub/datasets/halfbloodhamed/iranian-snack-and-chips-detection-yolo-format/versions/4/Iranian Snack and Chips Detection (YOLO Format)/train/images/20250506_211037_mp4-0060_jpg.rf.7dd7098690d1fbacf033b1072f7fffca.jpg', '/data/home/alan.arazi/.cache/kagglehub/datasets/halfbloodhamed/iranian-snack-and-chips-detection-yolo-format/versions/4/Iranian Snack and Chips Detection (YOLO Format)/train/images/20250506_205726_mp4-0153_jpg.rf.75df24fcd4f4f37d67ff049bdead788d.jpg', '/data/home/alan.arazi/.cache/kagglehub/datasets/halfbloodhamed/iranian-snack-and-chips-detection-yolo-format/versions/4/Iranian Snack and Chips Detection (YOLO Format)/train/images/20250506_205726_mp4-0108_jpg.rf.533e859a6bbff8ecd66cb7a559d8761e.jpg', '/data/home/alan.arazi/.cache/kagglehub/datasets/halfbloodhamed/iranian-snack-and-chips-detection-yolo-format/versions/4/Iranian Snack and Chips Detection (YOLO Format)/train/images/20250506_213503_mp4-0027_jpg.rf.8b5f0bc9a588aea8c78de0bb2e72f692.jpg', '/data/home/alan.arazi/.cache/kagglehub/datasets/halfbloodhamed/iranian-snack-and-chips-detection-yolo-format/versions/4/Iranian Snack and Chips Detection (YOLO Format)/train/images/20250506_205726_mp4-0007_jpg.rf.6248ee53825df1f871145c0d836ee805.jpg']
total snacks (int64, 8 distinct): ['3', '4', '2', '5', '1', '6', '0', '7']
distinct snacks (int64, 7 distinct): ['3', '2', '4', '1', '5', '0', '6']
'''

def load_df(dir_path: str) -> DataFrame:
    '''This is an object detection task, each image may has multiple objects. I randomly decided to convert it to a
    binary classification task: whether the image has at least one "Ashi Mashi snacks" (class id 0) or not.'''
    ret = []
    directories = [d for d in os.listdir(dir_path) if os.path.isdir(join(dir_path, d))]
    assert len(directories) == 1
    main_dir = directories[0]
    for split in ['train', 'valid', 'test']:
        split_dir = join(dir_path, main_dir, split)
        split_image_dir = join(split_dir, 'images')
        split_labels_dir = join(split_dir, 'labels')
        for img in os.listdir(split_image_dir):
            if not img.endswith('.jpg'):
                continue
            real_img_path = join(split_image_dir, img)
            label_path = join(split_labels_dir, img.replace('.jpg', '.txt'))
            objects = load_object_ids(label_path)
            total_objects = len(objects)
            distinct_objects = len(set(objects))
            has_ashi_mashi = bool(0 in objects)
            ret.append({IMAGE_FEATURE_NAME: real_img_path,
                        LABEL_NAME: has_ashi_mashi,
                        'total snacks': total_objects,
                        'distinct snacks': distinct_objects})
    df = DataFrame(ret)
    return df


def load_object_ids(txt_path: str) -> List[int]:
    ret = []
    label = load_txt(txt_path)
    for line in label.split('\n'):
        line = line.strip()
        if line == '':
            continue
        object_id = int(line.split()[0])
        ret.append(object_id)
    if ret:
        if min(ret) < 0 or max(ret) > 18:
            raise ValueError(f"Invalid object ids in {txt_path}: {ret}")
    return ret


CONTEXT = "Iranian Snack and Chips Detection Dataset"
TARGET = CuratedTarget(raw_name=LABEL_NAME, task_type=SupervisedTask.BINARY)
FEATURES = [CuratedFeature(raw_name=IMAGE_FEATURE_NAME, feat_type=FeatureType.IMAGE)]
IMAGE_FOLDER = ""
LOADING_FUNC = load_df
