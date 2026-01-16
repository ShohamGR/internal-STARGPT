import os
from os.path import join

from pandas import DataFrame

from tabstar_paper.datasets.curation_objects import CuratedTarget, CuratedFeature
from tabstar_paper.datasets.objects import SupervisedTask, FeatureType

LABEL_NAME = "spacecraft type"
IMAGE_FEATURE_NAME = "spacecraft image"

'''
Dataset Name: mightyglow/astronomical-image-and-csv-dataset/
====
Examples: 4902
====
URL: https://www.kaggle.com/mightyglow/astronomical-image-and-csv-dataset
====
Description: 
About Dataset
🪐 Astronomical Objects Dataset: 50,000 Images + Metadata
Welcome to the Astronomical Objects Dataset, a comprehensive collection of 50,000 high-resolution images of celestial bodies, including asteroids, planets, moons, stars, galaxies, nebulae, black holes, exoplanets, dwarf planets, constellations and stars and man-made space objects like rockets, space debris, spacecrafts (landers,orbiters,probes,rovers),satellites and space stations. This dataset is curated for AI/ML research, astronomy education, and image classification tasks in the realm of space exploration.

🌌 What's Inside?
📁 Images
50,000+ images organized by class (e.g., /asteroids, /planets, /moons, etc.)
Varying resolutions, optimized for computer vision models
Includes real astronomical captures from space missions and observatories
Also includes filtered images for Machine Learning Model, the filters are specific for astronomical images
📄 CSV Metadata Files (Column Descriptions)
For each category (e.g., asteroids, planets, etc.), there's a corresponding CSV file containing:

name: Common or scientific name of the object
description: Short summary about the object
distance_from_earth_km: Average distance from Earth
discovery_year: Year the object was discovered (if applicable)
diameter_km, mass_kg: Physical characteristics
fun_facts : Fun Facts about the specific object in that category
🤖 Python Bots
Image Rotator Bot - https://github.com/MightyGlow/Image-Rotator-Bot - This bot rotates an inputed image 2 degrees and saves each iteration in a single specified folder
File Renaming Bot - https://github.com/MightyGlow/File-Renaming-Bot - This bot renames each file in a entered folder in a numbered format. Eg: Earth_1, Earth_2
Filter Imaging Bot - https://github.com/MightyGlow/Filter-Imaging-Bot - This bot applies different filters to each image in the inputed folder and save it in the destinaation folder. There are various filters entered that you can choose from based on your preferences.
Star each of the above repository after using it to increase its reach so others can also apply it.
🔭 Use Cases
Training deep learning models for astronomical object classification
Building educational applications or interactive visualizations
Developing AI-powered space exploration tools
Supporting research in astrophysics and space sciences
🧠 Ideal For
Data scientists and machine learning practitioners
Astronomy enthusiasts and educators
Students working on space-related AI projects
Researchers building AI models for satellite or telescope data
📚 Upvote
Upvote this dataset for others to see and use it for their specific projects and purposes.

🚀 Let's Explore the Cosmos with AI!
Dive into the dataset and start building intelligent systems that can see and understand the universe like never before.
Let me know in the comments if you have more images that can be added to upgrade this dataset. I had to manually create this dataset as there are not many image datasets with specific astrnomical images for different categories each.

====
Target Variable: spacecraft type (object, 4 distinct): ['rovers', 'landers', 'orbiters', 'probes']
====
Features:

spacecraft image (object, 4902 distinct): ['/data/home/alan.arazi/.cache/kagglehub/datasets/mightyglow/astronomical-image-and-csv-dataset/versions/1/data/images/train/manmade objects/spacecraft/landers/landers_415.png', '/data/home/alan.arazi/.cache/kagglehub/datasets/mightyglow/astronomical-image-and-csv-dataset/versions/1/data/images/train/manmade objects/spacecraft/landers/landers_2.png', '/data/home/alan.arazi/.cache/kagglehub/datasets/mightyglow/astronomical-image-and-csv-dataset/versions/1/data/images/train/manmade objects/spacecraft/landers/landers_228.png', '/data/home/alan.arazi/.cache/kagglehub/datasets/mightyglow/astronomical-image-and-csv-dataset/versions/1/data/images/train/manmade objects/spacecraft/landers/landers_47.png', '/data/home/alan.arazi/.cache/kagglehub/datasets/mightyglow/astronomical-image-and-csv-dataset/versions/1/data/images/train/manmade objects/spacecraft/landers/landers_408.png', '/data/home/alan.arazi/.cache/kagglehub/datasets/mightyglow/astronomical-image-and-csv-dataset/versions/1/data/images/train/manmade objects/spacecraft/landers/landers_248.png', '/data/home/alan.arazi/.cache/kagglehub/datasets/mightyglow/astronomical-image-and-csv-dataset/versions/1/data/images/train/manmade objects/spacecraft/landers/landers_68.png', '/data/home/alan.arazi/.cache/kagglehub/datasets/mightyglow/astronomical-image-and-csv-dataset/versions/1/data/images/train/manmade objects/spacecraft/landers/landers_598.png', '/data/home/alan.arazi/.cache/kagglehub/datasets/mightyglow/astronomical-image-and-csv-dataset/versions/1/data/images/train/manmade objects/spacecraft/landers/landers_46.png', '/data/home/alan.arazi/.cache/kagglehub/datasets/mightyglow/astronomical-image-and-csv-dataset/versions/1/data/images/train/manmade objects/spacecraft/landers/landers_27.png']
'''

def load_df(dir_path: str) -> DataFrame:
    # There are so many images here. Let's focus only on types of spacecrafts
    ret = []
    img_dir_path = join(dir_path, IMAGE_FOLDER)
    for split in ['train', 'test']:
        spacecraft_dir = join(img_dir_path, split, "manmade objects/spacecraft")
        for spacecraft in ['landers', 'orbiters', 'probes',  'rovers']:
            dir_path = join(spacecraft_dir, spacecraft)
            for path in os.listdir(dir_path):
                relative_path = join(dir_path, path)
                ret.append({LABEL_NAME: spacecraft, IMAGE_FEATURE_NAME: relative_path})
    ret = DataFrame(ret)
    return ret


CONTEXT = "Astronomical Spacecraft Object Detection"
TARGET = CuratedTarget(raw_name=LABEL_NAME, task_type=SupervisedTask.MULTICLASS)
FEATURES = [CuratedFeature(raw_name=IMAGE_FEATURE_NAME, feat_type=FeatureType.IMAGE)]
IMAGE_FOLDER = "data/images"
LOADING_FUNC = load_df
