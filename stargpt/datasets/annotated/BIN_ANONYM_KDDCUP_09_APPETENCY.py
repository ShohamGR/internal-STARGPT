from tabstar_paper.datasets.curation_objects import CuratedFeature, CuratedTarget
from tabstar_paper.datasets.objects import SupervisedTask

'''
Dataset Name: kddcup09_appetency
====
Examples: 50000
====
URL: https://www.openml.org/search?type=data&id=46939
====
Description: This dataset was curated for [TabArena](https://tabarena.ai/) by the TabArena team
as part of the [TabArena Tabular ML IID Study](https://tabarena.ai/data-tabular-ml-iid-study).
For more details on the study, see our [paper](https://tabarena.ai/paper-tabular-ml-iid-study).

**Dataset Focus**: This dataset shall be used for evaluating predictive machine
learning models for independent and identically distributed tabular data. The
intended task is classification.

---
#### Dataset Metadata
- **Licence:** Public
- **Original Data Source:** https://kdd.org/kdd-cup/view/kdd-cup-2009/Intro
- **Reference (please cite)**: Guyon, Isabelle, et al. 'Analysis of the kdd cup 2009: Fast scoring on a large orange customer database.' KDD-Cup 2009 Competition. PMLR, 2009. https://proceedings.mlr.press/v7/guyon09.html
- **Dataset Year:** 2008
- **Dataset Description:** see the reference and the original data source for details.

#### Curation comments by the TabArena team (for code see the [page of the study](https://tabarena.ai/data-tabular-ml-iid-study)):
- We use the small (230 features) training data from the original data. 
- We use appetency as label.
- We dropped empty columns: "Var8", "Var15", "Var20", "Var31", "Var32", "Var39", "Var42", "Var48", "Var52", "Var55", "Var79", "Var141", "Var167", "Var169", "Var175", "Var185", "Var209", "Var230".
- Anomaly: this dataset has many missing values.
- Anomaly: the feature names and categorical values have no semantic meaning.
====

# This is the description of dataset: 1111

====
Description: **Author**:   
**Source**: Unknown - Date unknown  
**Please cite**:   

Datasets from ACM KDD Cup (http://www.sigkdd.org/kddcup/index.php)

KDD Cup 2009
http://www.kddcup-orange.com

Converted to ARFF format by TunedIT
Customer Relationship Management (CRM) is a key element of modern marketing strategies. The KDD Cup 2009 offers the opportunity to work on large marketing databases from the French Telecom company Orange to predict the propensity of customers to switch provider (churn), buy new products or services (appetency), or buy upgrades or add-ons proposed to them to make the sale more profitable (up-selling).
The most practical way, in a CRM system, to build knowledge on customer is to produce scores. A score (the output of a model) is an evaluation for all instances of a target variable to explain (i.e. churn, appetency or up-selling). Tools which produce scores allow to project, on a given population, quantifiable information. The score is computed using input variables which describe instances. Scores are then used by the information system (IS), for example, to personalize the customer relationship. An industrial customer analysis platform able to build prediction models with a very large number of input variables has been developed by Orange Labs. This platform implements several processing methods for instances and variables selection, prediction and indexation based on an efficient model combined with variable selection regularization and model averaging method. The main characteristic of this platform is its ability to scale on very large datasets with hundreds of thousands of instances and thousands of variables. The rapid and robust detection of the variables that have most contributed to the output prediction can be a key factor in a marketing application.
Appetency: In our context, the appetency is the propensity to buy a service or a product.
The training set contains 50,000 examples.
The first predictive 190 variables are numerical and the last 40 predictive variables are categorical.
The last target variable is binary {-1,1}.
====
Target Variable: appetency (category, 2 distinct): ['-1', '1']
====
Features:

Var1 (float64, 18 distinct): ['0.0', '8.0', '16.0', '24.0', '32.0', '40.0', '48.0', '56.0', '72.0', '128.0']
Var2 (float64, 2 distinct): ['0.0', '5.0']
Var3 (float64, 146 distinct): ['0.0', '3.0', '6.0', '144.0', '9.0', '18.0', '81.0', '99.0', '60.0', '90.0']
Var4 (float64, 4 distinct): ['0.0', '9.0', '18.0', '27.0']
Var5 (float64, 571 distinct): ['0.0', '432000.0', '864000.0', '3024000.0', '1296000.0', '2592000.0', '62160.0', '39580.0', '22395.0', '39020.0']
Var6 (float64, 1486 distinct): ['0.0', '777.0', '805.0', '798.0', '791.0', '812.0', '833.0', '784.0', '826.0', '840.0']
Var7 (float64, 8 distinct): ['7.0', '0.0', '14.0', '21.0', '28.0', '35.0', '140.0', '42.0']
Var9 (float64, 100 distinct): ['0.0', '8.0', '6.0', '10.0', '2.0', '4.0', '14.0', '12.0', '34.0', '16.0']
Var10 (float64, 534 distinct): ['0.0', '777600.0', '1555200.0', '2332800.0', '3110400.0', '3888000.0', '1425618.0', '5443200.0', '2709.0', '265230.0']
Var11 (float64, 5 distinct): ['8.0', '16.0', '24.0', '32.0', '40.0']
Var12 (float64, 22 distinct): ['0.0', '8.0', '16.0', '24.0', '32.0', '40.0', '48.0', '80.0', '104.0', '72.0']
Var13 (float64, 2634 distinct): ['0.0', '4.0', '8.0', '12.0', '16.0', '24.0', '20.0', '28.0', '36.0', '44.0']
Var14 (float64, 19 distinct): ['0.0', '2.0', '4.0', '8.0', '6.0', '16.0', '18.0', '20.0', '14.0', '12.0']
Var16 (float64, 597 distinct): ['0.0', '11.0', '11.32', '12.0', '11.64', '10.68', '49.08', '50.48', '47.68', '12.96']
Var17 (float64, 37 distinct): ['0.0', '5.0', '10.0', '15.0', '20.0', '35.0', '25.0', '30.0', '40.0', '45.0']
Var18 (float64, 26 distinct): ['0.0', '6.0', '12.0', '18.0', '24.0', '30.0', '36.0', '48.0', '42.0', '54.0']
Var19 (float64, 4 distinct): ['0.0', '9.0', '27.0', '18.0']
Var21 (float64, 734 distinct): ['132.0', '136.0', '0.0', '124.0', '128.0', '140.0', '144.0', '148.0', '152.0', '120.0']
Var22 (float64, 735 distinct): ['0.0', '165.0', '170.0', '155.0', '160.0', '175.0', '180.0', '185.0', '190.0', '150.0']
Var23 (float64, 29 distinct): ['0.0', '5.0', '10.0', '15.0', '20.0', '25.0', '30.0', '35.0', '40.0', '55.0']
Var24 (float64, 93 distinct): ['0.0', '2.0', '4.0', '6.0', '8.0', '10.0', '12.0', '14.0', '16.0', '18.0']
Var25 (float64, 271 distinct): ['0.0', '8.0', '16.0', '24.0', '32.0', '40.0', '48.0', '56.0', '64.0', '72.0']
Var26 (float64, 4 distinct): ['0.0', '3.0', '6.0', '9.0']
Var27 (float64, 3 distinct): ['0.0', '2.0', '4.0']
Var28 (float64, 4167 distinct): ['166.56', '220.08', '186.64', '253.52', '200.0', '133.12', '286.96', '233.44', '0.0', '320.4']
Var29 (float64, 2 distinct): ['0.0', '2.0']
Var30 (float64, 13 distinct): ['5.0', '0.0', '10.0', '15.0', '20.0', '25.0', '30.0', '45.0', '35.0', '60.0']
Var33 (float64, 298 distinct): ['0.0', '777600.0', '1512.0', '77211.0', '876159.0', '117774.0', '368037.0', '32796.0', '1849104.0', '165177.0']
Var34 (float64, 6 distinct): ['0.0', '8.0', '16.0', '24.0', '32.0', '56.0']
Var35 (float64, 13 distinct): ['0.0', '5.0', '10.0', '15.0', '20.0', '25.0', '30.0', '35.0', '40.0', '45.0']
Var36 (float64, 531 distinct): ['0.0', '345600.0', '172800.0', '518400.0', '1036800.0', '224.0', '27124.0', '684870.0', '83430.0', '1729950.0']
Var37 (float64, 550 distinct): ['0.0', '1555200.0', '712827.0', '712836.0', '712818.0', '3110400.0', '1425645.0', '5443200.0', '777600.0', '1555209.0']
Var38 (float64, 30832 distinct): ['0.0', '3628800.0', '7257600.0', '11404800.0', '518400.0', '1555200.0', '2073600.0', '10886400.0', '1036800.0', '4665600.0']
Var40 (float64, 27 distinct): ['0.0', '8.0', '16.0', '24.0', '32.0', '48.0', '40.0', '80.0', '56.0', '64.0']
Var41 (float64, 36 distinct): ['0.0', '7.0', '14.0', '21.0', '28.0', '35.0', '42.0', '56.0', '49.0', '77.0']
Var43 (float64, 20 distinct): ['0.0', '5.0', '10.0', '15.0', '20.0', '30.0', '40.0', '35.0', '25.0', '55.0']
Var44 (float64, 8 distinct): ['0.0', '9.0', '18.0', '27.0', '36.0', '63.0', '72.0', '135.0']
Var45 (float64, 343 distinct): ['2985.64', '5965.76', '839.84', '11994.5', '14412.84', '2664.96', '17507.62', '223.54', '1343.08', '1886.26']
Var46 (float64, 50 distinct): ['0.0', '8.0', '4.0', '12.0', '20.0', '16.0', '24.0', '28.0', '32.0', '44.0']
Var47 (float64, 11 distinct): ['0.0', '8.0', '16.0', '24.0', '32.0', '64.0', '48.0', '56.0', '80.0', '168.0']
Var49 (float64, 4 distinct): ['0.0', '6.0', '18.0', '12.0']
Var50 (float64, 63 distinct): ['0.0', '15.0', '20.0', '30.0', '5.0', '40.0', '45.0', '80.0', '35.0', '105.0']
Var51 (float64, 3561 distinct): ['0.0', '11520.0', '80640.0', '161280.0', '10560.16', '5627.28', '890.56', '217771.2', '182667.2', '1030.4']
Var53 (float64, 397 distinct): ['0.0', '777600.0', '3110400.0', '5443200.0', '2332800.0', '1555200.0', '6220800.0', '3888000.0', '712818.0', '10886400.0']
Var54 (float64, 5 distinct): ['0.0', '8.0', '16.0', '24.0', '32.0']
Var56 (float64, 195 distinct): ['0.0', '432000.0', '864000.0', '3105.0', '2192955.0', '143315.0', '124425.0', '17065.0', '194480.0', '161330.0']
Var57 (float64, 25614 distinct): ['2.4636', '0.0827', '4.0002', '6.078', '1.2123', '3.0998', '2.9013', '3.1797', '4.825', '3.7992']
Var58 (float64, 244 distinct): ['0.0', '691200.0', '1382400.0', '4838400.0', '218656.0', '217680.0', '68392.0', '511304.0', '120720.0', '72576.0']
Var59 (float64, 566 distinct): ['0.0', '777600.0', '5443200.0', '1890.0', '916659.0', '160948.8', '120301.2', '26541.0', '1008.0', '560717.1']
Var60 (float64, 47 distinct): ['0.0', '3.0', '6.0', '9.0', '12.0', '15.0', '18.0', '21.0', '27.0', '24.0']
Var61 (float64, 39 distinct): ['0.0', '8.0', '16.0', '24.0', '32.0', '40.0', '72.0', '48.0', '56.0', '80.0']
Var62 (float64, 12 distinct): ['0.0', '9.0', '18.0', '27.0', '45.0', '72.0', '36.0', '441.0', '288.0', '90.0']
Var63 (float64, 48 distinct): ['0.0', '6.0', '12.0', '24.0', '18.0', '36.0', '30.0', '42.0', '48.0', '60.0']
Var64 (float64, 233 distinct): ['12960.0', '0.0', '2441.88', '12006.18', '18140.04', '7574.94', '33481.98', '6451.2', '66928.77', '223728.3']
Var65 (float64, 15 distinct): ['9.0', '18.0', '27.0', '36.0', '45.0', '54.0', '63.0', '72.0', '81.0', '90.0']
Var66 (float64, 100 distinct): ['0.0', '16.0', '12.0', '20.0', '4.0', '8.0', '28.0', '24.0', '68.0', '44.0']
Var67 (float64, 2 distinct): ['0.0', '5.0']
Var68 (float64, 84 distinct): ['0.0', '42.0', '7.0', '28.0', '21.0', '35.0', '14.0', '49.0', '56.0', '63.0']
Var69 (float64, 838 distinct): ['0.0', '10886400.0', '2332800.0', '777600.0', '5443200.0', '8553600.0', '15055830.0', '5952294.0', '120888.0', '11452680.0']
Var70 (float64, 521 distinct): ['0.0', '1814400.0', '259200.0', '1296000.0', '518400.0', '3628800.0', '1555200.0', '2592000.0', '777600.0', '3888000.0']
Var71 (float64, 119 distinct): ['0.0', '24.0', '18.0', '12.0', '42.0', '30.0', '36.0', '54.0', '48.0', '84.0']
Var72 (float64, 8 distinct): ['3.0', '6.0', '9.0', '12.0', '15.0', '18.0', '21.0', '24.0']
Var73 (int64, 131 distinct): ['8', '16', '10', '34', '32', '12', '14', '18', '6', '28']
Var74 (float64, 371 distinct): ['0.0', '7.0', '14.0', '21.0', '28.0', '35.0', '42.0', '49.0', '56.0', '63.0']
Var75 (float64, 13 distinct): ['0.0', '5.0', '10.0', '15.0', '20.0', '25.0', '30.0', '40.0', '35.0', '45.0']
Var76 (float64, 29743 distinct): ['0.0', '1382400.0', '2764800.0', '5529600.0', '4838400.0', '4147200.0', '691200.0', '2073600.0', '3456000.0', '6220800.0']
Var77 (float64, 23 distinct): ['0.0', '6.0', '12.0', '18.0', '24.0', '36.0', '30.0', '42.0', '72.0', '48.0']
Var78 (float64, 13 distinct): ['0.0', '3.0', '6.0', '9.0', '12.0', '15.0', '18.0', '21.0', '24.0', '27.0']
Var80 (float64, 400 distinct): ['0.0', '259200.0', '518400.0', '777600.0', '237603.0', '475212.0', '62652.0', '49749.0', '2063274.0', '2757.0']
Var81 (float64, 43042 distinct): ['0.0', '259200.0', '33802.8', '191281.8', '183267.3', '30287.7', '49480.2', '221039.1', '57073.5', '171712.8']
Var82 (float64, 4 distinct): ['3.0', '0.0', '6.0', '9.0']
Var83 (float64, 195 distinct): ['0.0', '5.0', '10.0', '15.0', '20.0', '25.0', '35.0', '30.0', '45.0', '40.0']
Var84 (float64, 96 distinct): ['0.0', '4.0', '8.0', '16.0', '20.0', '40.0', '12.0', '36.0', '52.0', '44.0']
Var85 (float64, 149 distinct): ['0.0', '4.0', '2.0', '6.0', '8.0', '10.0', '12.0', '14.0', '16.0', '18.0']
Var86 (float64, 448 distinct): ['0.0', '1209600.0', '172800.0', '518400.0', '1555200.0', '691200.0', '2073600.0', '970.0', '928814.0', '198210.0']
Var87 (float64, 5 distinct): ['7.0', '0.0', '14.0', '28.0', '21.0']
Var88 (float64, 88 distinct): ['0.0', '10.0', '15.0', '20.0', '25.0', '5.0', '40.0', '30.0', '45.0', '35.0']
Var89 (float64, 16 distinct): ['0.0', '6.0', '12.0', '24.0', '18.0', '30.0', '42.0', '90.0', '72.0', '36.0']
Var90 (float64, 2 distinct): ['0.0', '7.0']
Var91 (float64, 119 distinct): ['0.0', '16.0', '12.0', '8.0', '28.0', '20.0', '24.0', '36.0', '32.0', '56.0']
Var92 (float64, 169 distinct): ['0.0', '8225.0', '596162.0', '2688.0', '406378.0', '304185.0', '113169.0', '31934.0', '7868.0', '729687.0']
Var93 (float64, 4 distinct): ['2.0', '4.0', '6.0', '8.0']
Var94 (float64, 20002 distinct): ['0.0', '36.0', '72.0', '108.0', '144.0', '216.0', '60.0', '84.0', '96.0', '24.0']
Var95 (float64, 267 distinct): ['0.0', '1036800.0', '518400.0', '1555200.0', '60894.0', '7590.0', '603606.0', '402270.0', '950538.0', '238050.0']
Var96 (float64, 33 distinct): ['0.0', '2.0', '4.0', '6.0', '8.0', '12.0', '10.0', '14.0', '16.0', '20.0']
Var97 (float64, 7 distinct): ['0.0', '6.0', '12.0', '18.0', '24.0', '30.0', '36.0']
Var98 (float64, 115 distinct): ['0.0', '345600.0', '4580.0', '117276.0', '163172.0', '44944.0', '179792.0', '316808.0', '42624.0', '108720.0']
Var99 (float64, 47 distinct): ['0.0', '8.0', '16.0', '24.0', '40.0', '32.0', '48.0', '56.0', '72.0', '80.0']
Var100 (float64, 5 distinct): ['0.0', '7.0', '14.0', '21.0', '28.0']
Var101 (float64, 28 distinct): ['0.0', '9.0', '18.0', '27.0', '36.0', '45.0', '54.0', '81.0', '72.0', '90.0']
Var102 (float64, 445 distinct): ['0.0', '22.5', '25920.0', '51840.0', '35249.13', '78417.54', '54939.15', '27772.11', '67556.07', '1200.78']
Var103 (float64, 39 distinct): ['0.0', '7.0', '14.0', '21.0', '35.0', '28.0', '70.0', '49.0', '42.0', '63.0']
Var104 (float64, 62 distinct): ['0.0', '9.0', '45.0', '54.0', '63.0', '36.0', '27.0', '72.0', '18.0', '90.0']
Var105 (float64, 62 distinct): ['0.0', '6.0', '30.0', '36.0', '42.0', '24.0', '18.0', '48.0', '12.0', '60.0']
Var106 (float64, 261 distinct): ['0.0', '259200.0', '518400.0', '1814400.0', '1036800.0', '9.0', '65352.0', '202431.0', '840954.0', '303024.0']
Var107 (float64, 24 distinct): ['0.0', '3.0', '6.0', '12.0', '9.0', '18.0', '15.0', '21.0', '27.0', '24.0']
Var108 (float64, 341 distinct): ['0.0', '345600.0', '691200.0', '1036800.0', '2419200.0', '2148.0', '1382400.0', '22632.0', '797520.0', '33960.0']
Var109 (float64, 209 distinct): ['32.0', '40.0', '24.0', '48.0', '0.0', '8.0', '16.0', '56.0', '64.0', '72.0']
Var110 (float64, 5 distinct): ['6.0', '12.0', '18.0', '30.0', '24.0']
Var111 (float64, 794 distinct): ['0.0', '518400.0', '3628800.0', '3628806.0', '414577.8', '44277.0', '334765.2', '14261.64', '610560.0', '3348.0']
Var112 (float64, 230 distinct): ['0.0', '16.0', '32.0', '24.0', '8.0', '40.0', '48.0', '64.0', '56.0', '72.0']
Var113 (float64, 48511 distinct): ['0.0', '45211.2', '-80196.8', '118363.6', '114611.6', '162338.0', '123069.2', '125842.4', '107644.4', '-680244.0']
Var114 (float64, 643 distinct): ['0.0', '1209600.0', '864000.0', '420.0', '357946.0', '2419200.0', '2818600.0', '1942368.0', '982448.0', '2281020.0']
Var115 (float64, 35 distinct): ['0.0', '18.0', '9.0', '27.0', '36.0', '45.0', '90.0', '63.0', '54.0', '81.0']
Var116 (float64, 2 distinct): ['0.0', '3.0']
Var117 (float64, 656 distinct): ['0.0', '345600.0', '172800.0', '1209600.0', '691200.0', '158410.0', '158406.0', '1404.0', '1209602.0', '864000.0']
Var118 (float64, 1 distinct): ['3.0']
Var119 (float64, 1487 distinct): ['0.0', '510.0', '520.0', '505.0', '500.0', '515.0', '525.0', '530.0', '495.0', '490.0']
Var120 (float64, 64 distinct): ['0.0', '6.0', '12.0', '18.0', '24.0', '30.0', '48.0', '42.0', '36.0', '54.0']
Var121 (float64, 33 distinct): ['0.0', '2.0', '4.0', '6.0', '8.0', '12.0', '10.0', '14.0', '18.0', '16.0']
Var122 (float64, 3 distinct): ['0.0', '3.0', '6.0']
Var123 (float64, 298 distinct): ['0.0', '6.0', '12.0', '18.0', '24.0', '30.0', '48.0', '54.0', '36.0', '60.0']
Var124 (float64, 347 distinct): ['0.0', '1382400.0', '2764800.0', '691200.0', '4838400.0', '2073600.0', '41104.0', '2041536.0', '1348792.0', '1940720.0']
Var125 (float64, 10505 distinct): ['0.0', '153.0', '171.0', '225.0', '117.0', '198.0', '423.0', '207.0', '261.0', '333.0']
Var126 (float64, 51 distinct): ['4.0', '-30.0', '-20.0', '-18.0', '-28.0', '-22.0', '-26.0', '-24.0', '6.0', '8.0']
Var127 (float64, 39 distinct): ['0.0', '8.0', '16.0', '32.0', '24.0', '40.0', '48.0', '56.0', '64.0', '80.0']
Var128 (float64, 88 distinct): ['0.0', '14.0', '21.0', '28.0', '35.0', '7.0', '56.0', '42.0', '63.0', '49.0']
Var129 (float64, 45 distinct): ['0.0', '2.0', '6.0', '4.0', '8.0', '10.0', '16.0', '12.0', '22.0', '18.0']
Var130 (float64, 2 distinct): ['0.0', '3.0']
Var131 (float64, 152 distinct): ['0.0', '6236344.0', '98052800.0', '460536.0', '5648.0', '74512.0', '253104.0', '30740240.0', '22552.0', '93384.0']
Var132 (float64, 19 distinct): ['0.0', '8.0', '16.0', '24.0', '32.0', '40.0', '48.0', '56.0', '64.0', '72.0']
Var133 (float64, 37603 distinct): ['0.0', '9504000.0', '3024000.0', '6048000.0', '1296000.0', '9072000.0', '8208000.0', '3888000.0', '12096000.0', '432000.0']
Var134 (float64, 33181 distinct): ['0.0', '518400.0', '1209600.0', '172800.0', '345600.0', '864000.0', '691200.0', '1036800.0', '1382400.0', '2419200.0']
Var135 (float64, 679 distinct): ['0.0', '18.69', '19.25', '22.12', '22.68', '18.13', '21.0', '21.56', '20.37', '216.86']
Var136 (float64, 534 distinct): ['0.0', '172800.0', '1209600.0', '484874.0', '74398.2', '436898.0', '328300.0', '106630.2', '88506.8', '35380.0']
Var137 (float64, 19 distinct): ['0.0', '4.0', '8.0', '12.0', '16.0', '20.0', '24.0', '28.0', '64.0', '36.0']
Var138 (float64, 2 distinct): ['0.0', '2.0']
Var139 (float64, 674 distinct): ['0.0', '1209600.0', '172800.0', '11678.0', '2419200.0', '345600.0', '518400.0', '19182.0', '26052.0', '226698.0']
Var140 (float64, 2648 distinct): ['0.0', '5.0', '10.0', '15.0', '20.0', '30.0', '40.0', '25.0', '65.0', '35.0']
Var142 (float64, 4 distinct): ['0.0', '4.0', '8.0', '12.0']
Var143 (float64, 4 distinct): ['0.0', '6.0', '12.0', '18.0']
Var144 (float64, 10 distinct): ['9.0', '0.0', '18.0', '27.0', '36.0', '45.0', '54.0', '63.0', '72.0', '81.0']
Var145 (float64, 88 distinct): ['0.0', '12.0', '18.0', '24.0', '30.0', '6.0', '48.0', '42.0', '54.0', '36.0']
Var146 (float64, 10 distinct): ['0.0', '8.0', '16.0', '24.0', '32.0', '40.0', '48.0', '80.0', '96.0', '64.0']
Var147 (float64, 5 distinct): ['2.0', '0.0', '4.0', '6.0', '8.0']
Var148 (float64, 119 distinct): ['0.0', '24.0', '16.0', '32.0', '56.0', '40.0', '48.0', '8.0', '72.0', '64.0']
Var149 (float64, 18652 distinct): ['0.0', '604800.0', '1209600.0', '554414.0', '1814400.0', '554421.0', '554435.0', '2419200.0', '554428.0', '554407.0']
Var150 (float64, 600 distinct): ['0.0', '864000.0', '396020.0', '432000.0', '396015.0', '396010.0', '396025.0', '1296000.0', '792030.0', '396005.0']
Var151 (float64, 19 distinct): ['0.0', '8.0', '16.0', '24.0', '32.0', '40.0', '56.0', '48.0', '64.0', '72.0']
Var152 (float64, 12 distinct): ['6.0', '0.0', '12.0', '18.0', '24.0', '30.0', '36.0', '42.0', '48.0', '66.0']
Var153 (float64, 36397 distinct): ['0.0', '10368000.0', '10713600.0', '9676800.0', '10022400.0', '10454680.0', '7257600.0', '10454800.0', '10455120.0', '10713640.0']
Var154 (float64, 388 distinct): ['0.0', '2073600.0', '4838400.0', '1382400.0', '691200.0', '9676800.0', '2764800.0', '3456000.0', '6220800.0', '597424.0']
Var155 (float64, 8 distinct): ['0.0', '5.0', '10.0', '15.0', '20.0', '25.0', '35.0', '30.0']
Var156 (float64, 100 distinct): ['0.0', '28.0', '21.0', '35.0', '7.0', '14.0', '49.0', '42.0', '119.0', '77.0']
Var157 (float64, 64 distinct): ['0.0', '4.0', '8.0', '16.0', '12.0', '24.0', '20.0', '28.0', '32.0', '36.0']
Var158 (float64, 18 distinct): ['0.0', '3.0', '6.0', '9.0', '12.0', '18.0', '21.0', '30.0', '45.0', '48.0']
Var159 (float64, 10 distinct): ['0.0', '9.0', '18.0', '27.0', '36.0', '54.0', '45.0', '63.0', '72.0', '99.0']
Var160 (float64, 402 distinct): ['0.0', '22.0', '2.0', '6.0', '4.0', '20.0', '18.0', '8.0', '16.0', '10.0']
Var161 (float64, 9 distinct): ['0.0', '9.0', '18.0', '27.0', '36.0', '45.0', '54.0', '63.0', '81.0']
Var162 (float64, 471 distinct): ['0.0', '777600.0', '1555200.0', '5443200.0', '2332800.0', '10886400.0', '3888000.0', '8541.0', '60642.0', '1518057.0']
Var163 (float64, 22957 distinct): ['0.0', '1036800.0', '2073600.0', '518400.0', '1555200.0', '3110400.0', '475212.0', '950424.0', '4147200.0', '3628800.0']
Var164 (float64, 19 distinct): ['0.0', '3.0', '6.0', '9.0', '12.0', '21.0', '15.0', '18.0', '27.0', '30.0']
Var165 (float64, 204 distinct): ['0.0', '172800.0', '345600.0', '25600.0', '30442.0', '26522.0', '8034.0', '30280.0', '167856.0', '12040.0']
Var166 (float64, 48 distinct): ['0.0', '7.0', '14.0', '21.0', '28.0', '35.0', '42.0', '63.0', '77.0', '49.0']
Var168 (float64, 453 distinct): ['247.84', '368.48', '358.16', '311.76', '283.36', '290.48', '301.52', '400.8', '322.72', '313.52']
Var170 (float64, 18 distinct): ['0.0', '3.0', '6.0', '9.0', '12.0', '15.0', '18.0', '21.0', '81.0', '51.0']
Var171 (float64, 746 distinct): ['0.0', '777600.0', '712827.0', '712845.0', '712822.5', '777604.5', '712840.5', '712831.5', '712818.0', '712849.5']
Var172 (float64, 13 distinct): ['7.0', '0.0', '14.0', '21.0', '28.0', '35.0', '42.0', '49.0', '56.0', '105.0']
Var173 (float64, 4 distinct): ['0.0', '2.0', '4.0', '6.0']
Var174 (float64, 29 distinct): ['0.0', '4.0', '8.0', '12.0', '16.0', '20.0', '24.0', '28.0', '32.0', '36.0']
Var176 (float64, 28 distinct): ['0.0', '4.0', '8.0', '12.0', '16.0', '64.0', '28.0', '52.0', '20.0', '68.0']
Var177 (float64, 443 distinct): ['0.0', '1209600.0', '604800.0', '2419200.0', '4233600.0', '1814400.0', '3024000.0', '59780.0', '96124.0', '2124549.0']
Var178 (float64, 30 distinct): ['0.0', '5.0', '10.0', '15.0', '20.0', '25.0', '40.0', '35.0', '30.0', '50.0']
Var179 (float64, 15 distinct): ['0.0', '5.0', '10.0', '15.0', '20.0', '25.0', '30.0', '40.0', '50.0', '890.0']
Var180 (float64, 547 distinct): ['0.0', '2419200.0', '4233600.0', '5598082.0', '5017544.0', '3343634.0', '8461460.0', '5468267.0', '9174900.0', '8043350.0']
Var181 (float64, 7 distinct): ['0.0', '7.0', '14.0', '21.0', '28.0', '49.0', '35.0']
Var182 (float64, 819 distinct): ['0.0', '1036800.0', '950430.0', '950436.0', '950460.0', '1036806.0', '950454.0', '950424.0', '950442.0', '2073600.0']
Var183 (float64, 374 distinct): ['0.0', '345600.0', '172800.0', '1209600.0', '158410.0', '138488.0', '50006.0', '518400.0', '234660.0', '174606.0']
Var184 (float64, 31 distinct): ['0.0', '4.0', '8.0', '12.0', '16.0', '20.0', '28.0', '24.0', '32.0', '36.0']
Var186 (float64, 13 distinct): ['0.0', '6.0', '12.0', '18.0', '24.0', '36.0', '30.0', '54.0', '42.0', '66.0']
Var187 (float64, 57 distinct): ['0.0', '2.0', '4.0', '10.0', '6.0', '8.0', '14.0', '12.0', '18.0', '16.0']
Var188 (float64, 535 distinct): ['15.6', '0.0', '16.14', '19.92', '16.68', '18.3', '18.84', '17.76', '19.38', '252.96']
Var189 (float64, 97 distinct): ['282.0', '276.0', '270.0', '288.0', '294.0', '252.0', '300.0', '264.0', '240.0', '246.0']
Var190 (float64, 328 distinct): ['0.0', '12960.0', '28114.92', '61643.61', '25953.57', '28484.01', '52025.85', '25952.94', '86645.08', '15517.53']
Var191 (category, 2 distinct): ['nan', 'r__I']
Var192 (category, 362 distinct): ['qFpmfo8zhV', 'DHeq9ayfAo', 'zKnr4RXktW', '8I1r4RXXnK', 'HYTrjIK12c', '75lr4RXktW', '1GdOj1KXzC', '2jirEyXktW', 'CxSr4RXktW', 'nan']
Var193 (category, 51 distinct): ['RO12', '2Knk1KF', 'AERks4l', 'g62hiBSaKg', 'e6CkoqApVR', 'LrdZy8QqgUfkVShG', 'rEUOq2QD1qfkRr6qpua', 'eSGpMwS8zSGgq_trOpckZ5', 'onTuEhrJJQy_H3IHkZku5AFczhYGqxJ890', 'w9ygS99Qp_']
Var194 (category, 4 distinct): ['nan', 'SEuy', 'lvza', 'CTUH']
Var195 (category, 23 distinct): ['taul', 'LfvqpCtLOY', 'CiJDdr4TQ0rGERIS', 'ev6I', 'CuXi4je', 'b_3Q', 'I9xt3GDRhUK7p', 'I9xt3GMcxUnBZ', 'I9xt3GBDKUbd8', 'ArtjQZ8ftr3NB']
Var196 (category, 4 distinct): ['1K8T', 'z3mO', 'JA1C', 'mKeq']
Var197 (category, 226 distinct): ['0Xwj', 'lK27', 'TyGl', '487l', 'JLbT', 'ssAy', 'z32l', 'kNzO', 'PGNs', '7gSz']
Var198 (category, 4291 distinct): ['fhk21Ss', 'PHNvXy8', 'iJzviRg', '9GJGgoz', '6CXYbuk', 'fqeOwLG', 'jCepSrJ', '_ybO0dd', 'pro8v8X', '0Vr7wZ4']
Var199 (category, 5074 distinct): ['r83_sZi', '_jTP8ioIlJ', 'FoJylxy', '76j2P_OLn0', 'glRBFJT8NN', 'msIqHk8toE', 'k10MzgT', 'hOpRIhsUSP', 'Tg7jjBB', '_UtlxbJ']
Var200 (category, 15416 distinct): ['nan', 'yP09M03', 'Uw6SDm8', 'Ipi9M03', 'EvCZGt8', 'MF5S0rA', '5YIkUea', 'b1M9M03', 'NvI9wLk', 'gWmZGt8']
Var201 (category, 3 distinct): ['nan', 'smXZ', '6dX3']
Var202 (category, 5714 distinct): ['nyZz', 'VNjO', '85IW', 'rlx_', 'gMVu', 'o5H_', 'Mx5G', 'xklU', '8FB6', '0xFv']
Var203 (category, 6 distinct): ['9_Y1', 'HLqf', 'F3hy', 'nan', 'dgxZ', 'pybr']
Var204 (category, 100 distinct): ['RVjC', 'k13i', 'm_h1', '7WNq', 'SkZj', 'rGJy', 'MBhA', 'RcM7', 'z5Ry', '15m3']
Var205 (category, 4 distinct): ['VpdQ', '09_Q', 'sJzTlal', 'nan']
Var206 (category, 22 distinct): ['IYzP', 'zm5i', 'nan', 'sYC_', 'haYg', 'hAFG', 'wMei', '43pnToF', 'kxE9', 'y6dw']
Var207 (category, 14 distinct): ['me75fM6ugJ', '7M47J5GA0pTYIFxg5uy', 'DHn_WUyBhW_whjA88g9bvA64_', 'Kxdu', 'NKv3VA1BpP', 'GjJ35utlTa_GNSvxxpb9ju', '6C53VA1kCv', '5iay', 'EBKcR3s6B22tD6gC36gm6S', '15TtzZrRt2']
Var208 (category, 3 distinct): ['kIsH', 'sBgB', 'nan']
Var210 (category, 6 distinct): ['uKAI', 'g5HH', '7A3j', 'oT7d', 'DM_V', '3av_']
Var211 (category, 2 distinct): ['L84s', 'Mtgm']
Var212 (category, 81 distinct): ['NhsEn4L', 'XfqtO3UdzaXh_', 'CrNX', 'Ie_5MZs', 'FMSzZ91zL2', '4kVnq_T26xq1p', 'h0lfDKh52u4GP', '_5OXC8MSLt', '9pUnzWLbztKTo', 'JBfYVit4g8']
Var213 (category, 2 distinct): ['nan', 'KdSa']
Var214 (category, 15416 distinct): ['nan', '5zARyjR', 'PXYoFMh', '5zA8Zov', 'PXYT2rL', '5zAF4eX', 'PXYoYU2', 'Fhz3CDZ', '5zA2v9l', 'PXYGJt6']
Var215 (category, 2 distinct): ['nan', 'eGzu']
Var216 (category, 2016 distinct): ['mAjbk_S', 'mAja5EA', 'kZJtVhC', 'XTbPUYD', 'beK4AFX', '11p4mKe', 'NGZxnJM', 'kZJyVg2', 'kq0aHkC', 'kq0n8Bj']
Var217 (category, 13991 distinct): ['nan', 'gvA6', '5smi', 'A1VJ', 'bru6', 's9FI', '4a9J', 'aINY', 'g2AX', 'JEC4']
Var218 (category, 3 distinct): ['cJvF', 'UYBR', 'nan']
Var219 (category, 23 distinct): ['FzaX', 'nan', 'AU8pNoi', 'qxDb', 'OFWH', 'AU8_WTd', 'wwPEXoilkr', 'Lmli', 'tdJW_Pm', 'FqMWi1g']
Var220 (category, 4291 distinct): ['4UxGlow', 'UF16siJ', 'ch2oGfM', 'Tvpip6Z', 'ROeipLp', 'fxJmel6', 'Oy_RPEi', 'L91KIiz', 'meWVy8V', 'XbZitea']
Var221 (category, 7 distinct): ['oslk', 'zCkv', 'd0EEeJi', 'QKW8DRm', 'Al6ZaUT', 'z4pH', 'JIiEFBU']
Var222 (category, 4291 distinct): ['catzS2D', 'APgdzOv', 'P6pu4Vl', 'hHJsvbM', 'K2SqEo9', 'WfsWw2A', 'FS4qjNq', 'CE7uk3u', 'DQ3u3MC', 'DHPNgqU']
Var223 (category, 5 distinct): ['LM8l689qOp', 'jySVZNlOJy', 'nan', 'M_8D', 'bCPvVye']
Var224 (category, 2 distinct): ['nan', '4n2X']
Var225 (category, 4 distinct): ['nan', 'ELof', 'kG3k', 'xG3x']
Var226 (category, 23 distinct): ['FSa2', 'Qu4f', 'WqMG', 'szEZ', '7P5s', 'fKCe', 'Aoh3', '5Acm', '453m', 'xb3V']
Var227 (category, 7 distinct): ['RAYp', 'ZI9m', '6fzt', '02N6s8f', 'nIGXDli', 'vJ_w8kB', 'nIGjgSB']
Var228 (category, 30 distinct): ['F2FyR07IdsN7I', '55YFVY9', 'ib5G6X1eUxUn6', 'R4y5gQQWY8OodqDV', 'xwM2aC7IdeMC0', 'TCU50_Yjmm6GIBZ0lL_', 'iyHGyLCEkQ', 'Zy3gnGM', 'F2FcTt7IdMT_v', 'SbOd7O8ky1wGNxp0Arj0Xs']
Var229 (category, 5 distinct): ['nan', 'am7c', 'mj86', 'sk2h', 'oJmt']
'''

CONTEXT = "CRM data from French Telecom company for product propensity prediction"
TARGET = CuratedTarget(raw_name="appetency", task_type=SupervisedTask.BINARY)