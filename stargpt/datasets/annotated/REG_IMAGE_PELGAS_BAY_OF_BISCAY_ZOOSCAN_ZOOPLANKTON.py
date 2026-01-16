import os
from os.path import join

from pandas import DataFrame

from tabstar2.utils.datasets import load_csv
from tabstar_paper.datasets.curation_objects import CuratedTarget, CuratedFeature
from tabstar_paper.datasets.objects import SupervisedTask, FeatureType


'''
Dataset Name: raghavdharwal/pelgas-bay-of-biscay-zooscan-zooplankton-dataset/
====
Examples: 100000
====
URL: https://www.kaggle.com/raghavdharwal/pelgas-bay-of-biscay-zooscan-zooplankton-dataset
====
Description:
title: PELGAS Bay of Biscay ZooScan zooplankton Dataset
subtitle: This dataset is of 1,153,507 zooplankton individuals images - 300 µm to 3.39 mm
keywords: ['earth and nature', 'earth science', 'computer vision', 'tabular', 'image', 'oceania']
licenses: [{'name': 'MIT'}]
description: # Introduction
This dataset is composed of 1,153,507 zooplankton individuals, zooplankton parts, non-living particles and imaging artefacts, ranging from 300 µm to 3.39 mm Equivalent Spherical Diameter, individually imaged and measured with the ZooScan (Gorsky et al., 2010). The objects were sorted in 127 taxonomic and morphological groups.
# Geography of dataset
The imaged objects originate from samples collected on the Bay of Biscay continental shelf, in spring, from 2004 to 2016 during the PELGAS ecosystemic surveys (Doray et al., 2018). The samples were collected with a WP2 200 µm mesh size fitted with a Hydrobios (back-run stop) mechanical flowmeter, generally from 100 m depth to the surface, or 5 m above the sea floor (if bottom depth less than 100 m) in vertical hauls, at night. From 2004 to 2006, vertical WP2 net tows were performed in the anchovy core distribution area in the southern Bay of Biscay and North of it until the Loire estuary only. Since 2009, WP2 sampling has been carried out at all PELGAS stations, up to the southern coast of Brittany. The samples were preserved in 4% buffered formaldehyde seawater solution directly after collection, until 2019-2020 where they were imaged with the ZooScan, in the lab, on land.

# Useful information for Feature Engineering

-- Each imaged object is geolocated, associated to a station, a cruise, a year and other metadata that enable the reconstruction of quantitative zooplankton communities for ecological studies (i.e. Grandrémy et al., 2023a).

-- Each object is described by 46 morphological and grey level based features (8 bits encoding, 0 = black, 255 = white), including size, automatically extracted on each individual image by the Zooprocess.

-- Each object was taxonomically identified using the web based application Ecotaxa with built-in, random forest and CNN based, semi-automatic sorting tools followed by expert validation or correction (Picheral et al., 2017).

-- This dataset is intended to be used for ecological studies as well as machine learning applied to plankton studies.

# The archive contains:

1. One tab separated file (PELGAS ZooScan zooplankton dataset) containing all data and metadata associated to each imaged and identified object. Metadata and features are in columns (n =71) and objects are in rows (n = 1,153,507).

2. One comma separated file containing the name, type, definition and unit of each field (column) in the .tsv (dataset_descriptor_zooscan).

3. One comma separated file containing the taxonomic list of the dataset, with counts and nature of the content of the category, i.e. “T” for taxonomical category, and “M” for morphological category (taxonomy_descriptor_zooscan).

4. A individual_images directory containing images of each imaged object sorted in subdirectories named according to objects’ identifications object_taxon appended to an Ecotaxa internal taxon numerical id classif_id (i.e. taxon__123456789) across years and sampling stations. Within subdirectories, each object is named after its unique internal Ecotaxa identifier, objid.

A Map of the sampling station location over the 2004-2016 period
====
Target Variable: object_slope (float64, 3786 distinct): ['0.069', '0.07', '0.071', '0.068', '0.067', '0.072', '0.066', '0.074', '0.077', '0.064']
====
Features:

object_lat (float64, 46 distinct): ['46.8653', '47.0825', '46.4036', '45.8353', '46.4311', '47.1975', '46.7789', '46.8178', '47.4683', '45.1508']
object_lon (float64, 46 distinct): ['-4.6947', '-5.0911', '-2.3994', '-2.1144', '-3.5472', '-3.4506', '-3.3281', '-5.0183', '-3.5689', '-1.4683']
object_date (int64, 23 distinct): ['20040520', '20040517', '20040503', '20040502', '20040516', '20040430', '20040509', '20040504', '20040519', '20040507']
object_time (int64, 45 distinct): ['235300', '200200', '21400', '213000', '10500', '4600', '210400', '223800', '231300', '22100']
object_depth_max (float64, 34 distinct): ['173.2051', '93.9693', '58.0237', '76.6044', '187.9385', '17.3205', '86.6025', '103.3662', '98.4808', '89.2708']
object_taxon (object, 102 distinct): ['detritus', 'artefact', 'Calanidae', 'Oithonidae', 'Calanoida', 'Noctiluca<Noctilucaceae', 'Oncaeidae', 'Temoridae', 'badfocus<artefact', 'fiber<detritus']
object_lineage (object, 102 distinct): ['not-living>detritus', 'not-living>artefact', 'living>Eukaryota>Opisthokonta>Holozoa>Metazoa>Arthropoda>Crustacea>Maxillopoda>Copepoda>Calanoida>Calanidae', 'living>Eukaryota>Opisthokonta>Holozoa>Metazoa>Arthropoda>Crustacea>Maxillopoda>Copepoda>Cyclopoida>Oithonidae', 'living>Eukaryota>Opisthokonta>Holozoa>Metazoa>Arthropoda>Crustacea>Maxillopoda>Copepoda>Calanoida', 'living>Eukaryota>Harosa>Alveolata>Myzozoa>Holodinophyta>Dinophyceae>Noctilucales>4>Noctilucaceae>Noctiluca', 'living>Eukaryota>Opisthokonta>Holozoa>Metazoa>Arthropoda>Crustacea>Maxillopoda>Copepoda>Poecilostomatoida>Oncaeidae', 'living>Eukaryota>Opisthokonta>Holozoa>Metazoa>Arthropoda>Crustacea>Maxillopoda>Copepoda>Calanoida>Temoridae', 'not-living>artefact>badfocus', 'not-living>detritus>fiber']
classif_id (int64, 102 distinct): ['84963', '85008', '61993', '62005', '45074', '58408', '78418', '61973', '85061', '85076']
object_area (int64, 18680 distinct): ['1059', '1082', '1056', '959', '646', '691', '998', '654', '1022', '923']
object_mean (float64, 13770 distinct): ['242.67', '242.69', '242.6', '242.56', '242.7', '242.61', '242.58', '242.73', '242.57', '242.52']
object_stddev (float64, 51690 distinct): ['1.543', '1.634', '1.624', '1.663', '1.578', '1.747', '1.695', '1.492', '1.766', '1.598']
object_mode (int64, 238 distinct): ['243', '242', '241', '240', '239', '238', '237', '236', '235', '234']
object_min (int64, 237 distinct): ['237', '238', '236', '77', '83', '81', '80', '79', '85', '84']
object_max (int64, 14 distinct): ['243', '255', '249', '248', '247', '246', '245', '250', '244', '251']
object_perim. (float64, 42561 distinct): ['174.65', '174.31', '178.89', '175.72', '181.97', '159.24', '174.07', '179.72', '172.65', '175.14']
object_major (float64, 4669 distinct): ['57.5', '61.8', '62.3', '57.3', '58.2', '56.7', '59.3', '54.9', '60.6', '60.4']
object_minor (float64, 2177 distinct): ['24.1', '23.5', '24.5', '24.3', '23.9', '25.7', '25.3', '24.2', '23.8', '26.4']
object_circ. (float64, 916 distinct): ['0.012', '0.014', '0.013', '0.01', '0.011', '0.015', '0.101', '0.104', '0.118', '0.113']
object_feret (float64, 5980 distinct): ['67.1', '63.6', '62.4', '65.8', '60.4', '62.6', '64.4', '69.4', '59.5', '67.7']
object_intden (int64, 93749 distinct): ['159752', '165953', '171631', '166374', '153209', '500947', '172077', '195820', '180659', '179823']
object_median (int64, 225 distinct): ['243', '237', '238', '236', '235', '234', '239', '233', '232', '231']
object_skew (float64, 5952 distinct): ['0.212', '0.227', '0.146', '0.218', '0.311', '0.104', '0.274', '0.176', '0.236', '0.046']
object_kurt (float64, 12389 distinct): ['-1.52', '-1.543', '-1.508', '-1.475', '-1.515', '-1.529', '-1.494', '-1.46', '-1.526', '-1.491']
object_%area (float64, 4070 distinct): ['0.0', '0.08', '0.09', '0.04', '0.07', '0.05', '0.1', '0.06', '0.03', '0.02']
object_area_exc (int64, 18438 distinct): ['796', '1048', '1056', '1046', '1087', '971', '1084', '1022', '1092', '1049']
object_fractal (float64, 779 distinct): ['1.193', '1.206', '1.181', '1.179', '1.197', '1.182', '1.174', '1.194', '1.195', '1.184']
object_skelarea (int64, 4267 distinct): ['102', '94', '101', '84', '93', '106', '92', '99', '119', '81']
object_histcum1 (int64, 224 distinct): ['240', '241', '224', '229', '230', '231', '223', '228', '225', '222']
object_histcum2 (int64, 215 distinct): ['241', '236', '233', '235', '234', '232', '237', '231', '230', '229']
object_histcum3 (int64, 176 distinct): ['242', '239', '238', '240', '237', '236', '235', '234', '233', '232']
object_nb1 (int64, 90 distinct): ['1', '2', '3', '0', '4', '5', '6', '7', '8', '9']
object_nb2 (int64, 107 distinct): ['1', '2', '3', '0', '4', '5', '6', '7', '8', '9']
object_symetrieh (float64, 11973 distinct): ['3.001', '3.032', '2.829', '3.166', '2.743', '3.021', '2.948', '2.903', '3.025', '2.84']
object_symetriev (float64, 12002 distinct): ['3.153', '2.592', '2.836', '3.016', '3.135', '3.335', '2.982', '2.008', '3.117', '2.717']
object_symetriehc (int64, 82 distinct): ['3', '4', '2', '5', '6', '7', '8', '9', '10', '11']
object_symetrievc (int64, 83 distinct): ['3', '4', '2', '5', '6', '7', '8', '9', '10', '11']
object_convperim (int64, 2319 distinct): ['192', '194', '186', '198', '188', '196', '202', '206', '200', '184']
object_convarea (int64, 25675 distinct): ['1305', '1452', '1430', '1379', '1311', '1425', '1394', '1315', '1303', '1266']
object_fcons (float64, 66933 distinct): ['0.0', '1.434', '1.354', '1.545', '0.675', '1.381', '1.063', '0.559', '1.064', '0.645']
object_thickr (float64, 6419 distinct): ['2.0', '3.0', '2.5', '2.333', '2.667', '2.4', '2.25', '3.5', '4.0', '2.8']
object_esd (float64, 19367 distinct): ['34.9433', '36.668', '36.72', '37.1166', '29.6616', '28.8565', '28.6795', '35.6468', '34.2812', '36.1434']
object_elongation (float64, 12486 distinct): ['2.0', '1.5', '3.0', '2.5', '1.6667', '2.3333', '1.3333', '2.6667', '1.75', '2.25']
object_range (int64, 246 distinct): ['12', '11', '13', '10', '160', '159', '164', '157', '155', '162']
object_meanpos (float64, 82142 distinct): ['-1.2727', '-1.2222', '-1.0833', '-1.3256', '-1.0', '-1.1739', '-1.1277', '-1.1505', '-1.1053', '-1.0408']
object_centroids (float64, 379 distinct): ['0.0', '1.0', '1.4142', '2.2361', '2.0', '3.1623', '2.8284', '3.6056', '3.0', '5.0']
object_cv (float64, 72209 distinct): ['0.8242', '0.8241', '0.8244', '0.8245', '0.8245', '0.8244', '0.8247', '0.8246', '0.8244', '0.8245']
object_sr (float64, 4035 distinct): ['16.6667', '20.0', '33.3333', '14.2857', '15.3846', '18.1818', '10.0', '12.5', '25.0', '22.2222']
object_perimareaexc (float64, 94924 distinct): ['5.0536', '5.2915', '5.3072', '6.2554', '6.0', '5.016', '5.3666', '5.2026', '9.0', '5.446']
object_feretareaexc (float64, 80968 distinct): ['2.0', '1.7321', '2.1213', '2.4962', '1.633', '1.7678', '2.4749', '1.7889', '2.0303', '2.4495']
object_perimferet (float64, 40135 distinct): ['3.0', '4.0', '3.5', '3.3333', '2.6667', '3.6667', '5.0', '2.75', '3.1667', '3.25']
object_perimmajor (float64, 40349 distinct): ['3.0', '4.0', '5.0', '3.5', '3.3333', '3.25', '6.0', '3.2', '4.5', '3.6667']
object_circex (float64, 94912 distinct): ['0.4488', '0.492', '0.1551', '0.4461', '0.4237', '0.4643', '0.1954', '0.3211', '0.3491', '0.2513']
object_cdexc (float64, 31782 distinct): ['0.0', '0.0304', '0.0342', '0.0319', '0.0314', '0.0317', '0.0308', '0.0379', '0.032', '0.0309']
sample_id (object, 46 distinct): ['pelgas2004_i0319_wp2', 'pelgas2004_i0329_wp2', 'pelgas2004_i0268_wp2', 'pelgas2004_i0252_wp2', 'pelgas2004_i0281_wp2', 'pelgas2004_i0303_wp2', 'pelgas2004_i0297_wp2', 'pelgas2004_i0328_wp2', 'pelgas2004_i0312_wp2', 'pelgas2004_i0232_wp2']
sample_stationid (object, 46 distinct): ['i0319', 'i0329', 'i0268', 'i0252', 'i0281', 'i0303', 'i0297', 'i0328', 'i0312', 'i0232']
sample_bottomdepth (float64, 46 distinct): ['139.8', '137.9', '24.83', '74.95', '106.9', '71.7', '101.05', '216.0', '45.4', '29.9']
sample_tot_vol (float64, 25 distinct): ['25.0', '50.0', '5.0', '16.75', '10.75', '27.5', '23.75', '6.25', '8.75', '24.0']
acq_min_mesh (int64, 2 distinct): ['200', '1000']
acq_max_mesh (int64, 2 distinct): ['1000', '999999']
acq_sub_part (float64, 9 distinct): ['32.0', '64.0', '8.0', '16.0', '128.0', '1.0', '4.0', '2.0', '256.0']
object_image (object, 100000 distinct): ['fiber_detritus__85076/334803779.jpg', 'Appendicularia__85123/334805377.jpg', 'Appendicularia__85123/334805472.jpg', 'Appendicularia__85123/334805575.jpg', 'Appendicularia__85123/334805277.jpg', 'Cladocera__45036/334779786.jpg', 'Appendicularia__85123/334805473.jpg', 'Calanidae__61993/334787176.jpg', 'fiber_detritus__85076/334804090.jpg', 'multiple_other__85079/334804676.jpg']
'''

OBJECT_IMAGE = "object_image"

def load_df(dir_path: str) -> DataFrame:
    # Too big of a file. Just take 100K deterministicly
    df = load_csv(dir_path, "101138.tsv", sep="\t")
    df = df[:100000]
    df = collect_all_images(df, dir_path)
    return df


def collect_all_images(df, dir_path: str):
    objid2path = {}
    img_folder = join(dir_path, IMAGE_FOLDER)
    for type_dir in os.listdir(img_folder):
        for img_file in os.listdir(join(img_folder, type_dir)):
            if not img_file.endswith(".jpg"):
                continue
            objid = int(img_file.replace(".jpg", ""))
            objid2path[objid] = join(type_dir, img_file)

    df[OBJECT_IMAGE] = df["objid"].map(objid2path)
    return df


CONTEXT = ""
TARGET = CuratedTarget(raw_name='object_slope', task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = ["object_id", "objid", "object_depth_min", "process_particle_pixel_size_mm",
                "sample_comment", "sample_net_type", "sample_ship", "sample_program", "process_id", "acq_id"]
FEATURES = [CuratedFeature(raw_name=OBJECT_IMAGE, feat_type=FeatureType.IMAGE)]
IMAGE_FOLDER = "101141/individual_images"
LOADING_FUNC = load_df
