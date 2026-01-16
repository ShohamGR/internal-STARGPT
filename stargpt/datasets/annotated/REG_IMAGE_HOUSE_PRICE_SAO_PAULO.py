from collections import defaultdict
import os
from os.path import join
from typing import Dict, List

from pandas import DataFrame, read_csv

from tabstar2.utils.datasets import load_csv
from tabstar_paper.datasets.curation_objects import CuratedTarget, CuratedFeature
from tabstar_paper.datasets.objects import SupervisedTask, FeatureType


'''
Dataset Name: rogerio/image-based-property-price-prediction-in-so-paulo/
====
Examples: 23433
====
URL: https://www.kaggle.com/rogerio/image-based-property-price-prediction-in-so-paulo
====
Description: 
Image-Based Property Price Prediction in São Paulo

About Dataset
This dataset provides data for advanced Property Price Prediction. It contains data from 23,433 properties for sale in the city of São Paulo (Brazil) advertised during April 2024. The data includes the price, over 30 property characteristics (m2, number of bedrooms, garage, etc.) and images of the listings, for a total of 407,567 images.

====
Target Variable: price (int64, 4674 distinct): ['1200000', '750000', '850000', '1300000', '550000', '1600000', '1100000', '650000', '450000', '950000']
====
Features:

title (object, 15182 distinct): ['Imóvel a Venda', 'São Paulo - Apartamento Padrão - Tatuapé', 'São Paulo - Apartamento Padrão - Mooca', 'Apartamento com 3 quartos à venda em Aclimação - SP', 'Casa com 3 quartos à venda em Aclimação - SP', 'APARTAMENTO - ACLIMAÇÃO - SP', 'SãO PAULO - Apartamento Padrão - Bela Vista', 'Apartamento com 4 quartos à venda em Aclimação - SP', 'São Paulo - Casa Padrão - Vila Prudente', 'Apartamento à venda, Barra Funda, São Paulo, SP']
location (object, 941 distinct): ['São Paulo, Bela Vista', 'São Paulo, Santa Cecília', 'São Paulo, Perdizes', 'São Paulo, Pinheiros', 'São Paulo, Aclimação', 'São Paulo, Consolação', 'São Paulo, Santana', 'São Paulo, Barra Funda', 'São Paulo, Cambuci', 'São Paulo, Mooca']
date (object, 13 distinct): ['24/04/2024', '25/04/2024', '22/04/2024', '23/04/2024', '20/04/2024', '21/04/2024', '19/04/2024', '14/04/2024', '15/04/2024', '13/04/2024']
destaque (bool, 2 distinct): ['0', '1']
Categoria (object, 2 distinct): ['Apartamentos', 'Casas']
Tipo (object, 9 distinct): ['Padrão', 'Cobertura', 'Casa de condomínio', 'Loft', 'Kitnet', 'Casa de vila', 'Duplex ou triplex', 'Venda - casa em rua pública', 'Venda - casa em vila']
Condomínio (float64, 2310 distinct, 0.7% missing): ['0.0', '1.0', '1200.0', '1500.0', '1100.0', '1000.0', '600.0', '1300.0', '800.0', '900.0']
Área útil (int64, 596 distinct): ['70', '120', '110', '60', '100', '50', '150', '80', '130', '140']
Quartos (object, 6 distinct): ['3', '2', '4', '1', '5 ou mais', '0']
Banheiros (object, 5 distinct, 0.0% missing): ['2', '1', '3', '4', '5 ou mais']
Vagas na garagem (object, 6 distinct): ['1', '2', '0', '3', '4', '5 ou mais']
Detalhes do imóvel (object, 407 distinct, 30.4% missing): ['Área de serviço', 'Churrasqueira', 'Churrasqueira, Piscina', 'Academia, Churrasqueira, Piscina', 'Piscina', 'Área de serviço, Churrasqueira, Piscina', 'Academia, Piscina', 'Varanda', 'Área de serviço, Churrasqueira', 'Área de serviço, Varanda']
Detalhes do condomínio (object, 233 distinct, 30.3% missing): ['Permitido animais', 'Piscina, Salão de festas', 'Permitido animais, Salão de festas', 'Piscina', 'Academia, Piscina, Salão de festas', 'Salão de festas', 'Academia, Condomínio fechado, Elevador, Permitido animais, Piscina, Portaria, Salão de festas', 'Academia, Elevador, Permitido animais, Piscina, Salão de festas', 'Elevador, Permitido animais', 'Academia, Salão de festas']
IPTU (float64, 2145 distinct, 0.9% missing): ['0.0', '1.0', '100.0', '200.0', '300.0', '150.0', '250.0', '500.0', '50.0', '350.0']
ZONA (object, 5 distinct): ['CENTRO', 'NORTE', 'OESTE', 'LESTE', 'SUL']
Academia (int64, 2 distinct): ['0', '1']
Elevador (int64, 2 distinct): ['0', '1']
Permitido animais (int64, 2 distinct): ['0', '1']
Piscina (int64, 2 distinct): ['0', '1']
Portaria (int64, 2 distinct): ['0', '1']
Salão de festas (int64, 2 distinct): ['0', '1']
Condomínio fechado (int64, 2 distinct): ['0', '1']
Segurança 24h (int64, 2 distinct): ['0', '1']
Portão eletrônico (int64, 2 distinct): ['0', '1']
Área murada (int64, 2 distinct): ['0', '1']
Área de serviço (int64, 2 distinct): ['0', '1']
Armários na cozinha (int64, 2 distinct): ['0', '1']
Armários no quarto (int64, 2 distinct): ['0', '1']
Churrasqueira (int64, 2 distinct): ['0', '1']
Mobiliado (int64, 2 distinct): ['0', '1']
Quarto de serviço (int64, 2 distinct): ['0', '1']
Ar condicionado (int64, 2 distinct): ['0', '1']
Porteiro 24h (int64, 2 distinct): ['0', '1']
Varanda (int64, 2 distinct): ['0', '1']
bairro (object, 912 distinct): ['Bela Vista', 'Santa Cecília', 'Perdizes', 'Pinheiros', 'Aclimação', 'Consolação', 'Santana', 'Barra Funda', 'Cambuci', 'Mooca']
image_cnt (int64, 34 distinct): ['20', '40', '18', '16', '14', '15', '19', '36', '17', '38']
first_pic (object, 23433 distinct): ['Imagens_05/0_4.png', 'Imagens_05/1_8.png', 'Imagens_05/2_6.png', 'Imagens_05/3_2.png', 'Imagens_05/4_16.png', 'Imagens_05/5_12.png', 'Imagens_05/6_1.png', 'Imagens_05/7_2.png', 'Imagens_05/8_16.png', 'Imagens_05/9_6.png']
'''

INDEX = "index"
FIRST_PIC = "first_pic"

def load_df(dir_path: str) -> DataFrame:
    df = load_csv(dir_path, "data.csv", sep=";")
    collect_all_images(df, dir_path=dir_path)
    return df



def collect_all_images(df: DataFrame, dir_path: str):
    idx2images = defaultdict(list)
    img_folder = join(dir_path, IMAGE_FOLDER)
    for dir_num in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14']:
        dir_folder = f'Imagens_{dir_num}'
        for img_file in os.listdir(join(img_folder, dir_folder)):
            # format is '3187_20.png'
            img_idx, img_num = img_file.split('_')
            assert img_num.endswith('.png')
            img_idx = int(img_idx)
            img_path = join(dir_folder, img_file)
            idx2images[img_idx].append(img_path)
    df[INDEX] = df[INDEX].apply(lambda x: idx2images[x])
    df['image_cnt'] = df[INDEX].apply(lambda x: len(x))
    df[FIRST_PIC] = df[INDEX].apply(lambda x: x[0])



CONTEXT = ""
TARGET = CuratedTarget(raw_name='price', task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = [INDEX, 'oldPrice']
TEXT_FEATURES = [CuratedFeature(raw_name=f, feat_type=FeatureType.TEXT) for f in 
['Detalhes do condomínio', 'Detalhes do imóvel', 'bairro', 'location', 'title']
]
FEATURES = [CuratedFeature(raw_name=FIRST_PIC, feat_type=FeatureType.IMAGE)] + TEXT_FEATURES
IMAGE_FOLDER = "Imagens"
LOADING_FUNC = load_df
