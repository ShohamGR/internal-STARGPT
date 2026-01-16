from tabstar_paper.datasets.curation_objects import CuratedTarget, CuratedFeature
from tabstar_paper.datasets.objects import SupervisedTask, FeatureType

'''
Dataset Name: REG_IMAGE_CD18_CELLPHONE_PRICE_PREDICTION
====
Examples: 3165
====
URL: https://github.com/AidinZe/CD18-Cellphone-Dataset-with-18-Features
====
Description:
Multimodal_price_prediction_dataset
dataset of Multimodal price prediction article

DOI = 10.1007/s40745-021-00326-z

Article's Abstract:
Price prediction is one of the examples related to forecasting tasks and is a project based on data science. Price prediction analyzes data and predicts the cost of new products. The goal of this research is to achieve an arrangement to predict the price of a cellphone based on its specifications. So, five deep learning models are proposed to predict the price range of a cellphone, one unimodal and four multimodal approaches. The multimodal methods predict the prices based on the graphical and non-graphical features of cellphones that have an important effect on their valorizations. Also, to evaluate the efficiency of the proposed methods, a cellphone dataset has been gathered from GSMArena. The experimental results show 88.3% F1-score, which confirms that multimodal learning leads to more accurate predictions than state-of-the-art techniques.

Cite this article
Zehtab-Salmasi, A., Feizi-Derakhshi, AR., Nikzad-Khasmakhi, N. et al. Multimodal Price Prediction. Ann. Data. Sci. (2021). https://doi.org/10.1007/s40745-021-00326-z
====
Target Variable: Price (float64, 391 distinct): ['200.0', '150.0', '100.0', '130.0', '120.0', '250.0', '170.0', '110.0', '180.0', '300.0']
====
Features:

name (object, 78 distinct): ['Samsung', 'HTC', 'LG', 'Huawei', 'Lenovo', 'Motorola', 'ZTE', 'Sony', 'Asus', 'Oppo']
Model (object, 3106 distinct): ['3', '7', '2', 'Z3', 'A7', 'K3', 'A1', 'F5', 'X', '6']
Release Date (int64, 17 distinct): ['2014', '2015', '2018', '2017', '2013', '2016', '2012', '2019', '2011', '2010']
Weigth (float64, 564 distinct): ['150.0', '145.0', '140.0', '160.0', '155.0', '130.0', '135.0', '170.0', '165.0', '120.0']
os (object, 20 distinct): ['Android', 'Microsoft', 'iOS', 'Feature', 'Tizen', 'BlackBerry', 'watchOS', 'Firefox', 'Customized', 'KaiOS']
Storage (float64, 45 distinct): ['16384.0', '8192.0', '32768.0', '4096.0', '65536.0', '131072.0', '512.0', '256.0', '1024.0', '64.0']
hit (float64, 79 distinct): ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']
hit_count (int64, 3164 distinct): ['67944', '493024', '315823', '487475', '588987', '298010', '2598838', '263289', '127339', '1494147']
display_size (float64, 150 distinct): ['5.0', '5.5', '4.0', '4.5', '5.2', '4.3', '6.0', '5.7', '4.7', '7.0']
V_resolution (int64, 40 distinct): ['720', '1080', '480', '540', '320', '1440', '800', '240', '600', '1200']
H_resolution (int64, 76 distinct): ['1280', '1920', '800', '960', '854', '480', '2560', '1440', '2340', '320']
camera (float64, 29 distinct): ['13.0', '8.0', '5.0', '16.0', '3.0', '12.0', '2.0', '20.0', '48.0', '0.0']
video (int64, 14 distinct): ['1080', '720', '2160', '144', '480', '0', '240', '320', '288', '448']
prcessor (object, 31 distinct): ['Snapdragon', 'mediaTek', 'Exynos', 'TI', 'HiSilicon', 'Intel', 'Nvidia', 'MSM', 'Apple', 'Spreadtrum']
ram (int64, 27 distinct): ['1024', '2048', '3072', '512', '4096', '6144', '256', '1536', '768', '64']
Battery (int64, 353 distinct): ['3000', '4000', '2000', '1500', '2500', '2600', '3500', '2100', '1800', '2300']
Battery_type (object, 2 distinct): ['Li-Ion', 'Li-Po']
Picture (object, 2964 distinct): ['samsung-galaxy-s-4-i9500-black-mist.jpg', 'htc-one-m8.jpg', 'apple-ipad-3-new.jpg', 'lg-g2-mini-ofic1.jpg', 'vivo-v9-.jpg', 'samsung-galaxy-tab-4-70.jpg', 'samsung-galaxy-s5-g900f.jpg', 'samsung-galaxy-s6-edge-plus.jpg', 'asus-zenfone-4-max-zc554kl.jpg', 'samsung-galaxy-note5.jpg']
'''

def trim_image_path_prefix(img_path: str) -> str:
    return img_path.replace('https://fdn2.gsmarena.com/vv/bigpic/', '')


CONTEXT = ""
TARGET = CuratedTarget(raw_name="Price", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name="Picture", feat_type=FeatureType.IMAGE, processing_func=trim_image_path_prefix),
            CuratedFeature(raw_name="Model", feat_type=FeatureType.TEXT),]
IMAGE_FOLDER = "cd18_images"
