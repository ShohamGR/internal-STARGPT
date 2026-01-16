import os
from os.path import join

from pandas import DataFrame

from tabstar_paper.datasets.curation_objects import CuratedTarget, CuratedFeature
from tabstar_paper.datasets.objects import SupervisedTask, FeatureType

LABEL_NAME = "Defect"
IMAGE_FEATURE_NAME = "Magnetic Tile"

'''
Dataset Name: alex000kim/magnetic-tile-surface-defects/
====
Examples: 2688
====
URL: https://www.kaggle.com/alex000kim/magnetic-tile-surface-defects
====
Description:
About Dataset
Surface defect detection is a core process of filtering unqualified products, however, the procedure can rarely be finished automatically. It is recorded that almost three-quarters of workers are employed to inspect product quality at the magnetic tile factories in Zhejiang Province, China, the largest magnetic tile production base in the world. To relieve human labor, many image processing techniques have been proposed to attempt such examination tasks. There have been several bottlenecks presented in the automatic damage detection for magnetic tiles, including the complexity of texture, the variety of defect shape, and the randomness of illumination conditions on magnetic tiles. The target defects such as blowhole, crack, break, fray are shown in Figure below:
====
Target Variable: Defect (object, 6 distinct): ['MT_Free', 'MT_Blowhole', 'MT_Uneven', 'MT_Break', 'MT_Crack', 'MT_Fray']
====
Features:

Magnetic Tile (object, 2688 distinct): ['MT_Blowhole/Imgs/exp1_num_108719.jpg', 'MT_Blowhole/Imgs/exp1_num_108719.png', 'MT_Blowhole/Imgs/exp1_num_108889.jpg', 'MT_Blowhole/Imgs/exp1_num_108889.png', 'MT_Blowhole/Imgs/exp1_num_262480.jpg', 'MT_Blowhole/Imgs/exp1_num_262480.png', 'MT_Blowhole/Imgs/exp1_num_265077.jpg', 'MT_Blowhole/Imgs/exp1_num_265077.png', 'MT_Blowhole/Imgs/exp1_num_290998.jpg', 'MT_Blowhole/Imgs/exp1_num_290998.png']
'''

def load_magnetic_tile_df(dir_path: str) -> DataFrame:
    ret = []
    images_path = join(dir_path, IMAGE_FOLDER)
    for magnetic_type in sorted(os.listdir(images_path)):
        magnetic_type_path = join(magnetic_type, "Imgs")
        if not os.path.isdir(join(images_path, magnetic_type_path)):
            continue
        for img_name in sorted(os.listdir(join(images_path, magnetic_type_path))):
            img_path = join(magnetic_type_path, img_name)
            ret.append({IMAGE_FEATURE_NAME: img_path, LABEL_NAME: magnetic_type})
    ret = DataFrame(ret)
    return ret


CONTEXT = "Object recognition of magnetic tile defects"
TARGET = CuratedTarget(raw_name=LABEL_NAME, task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name=IMAGE_FEATURE_NAME, feat_type=FeatureType.IMAGE)]
IMAGE_FOLDER = ""
LOADING_FUNC = load_magnetic_tile_df
