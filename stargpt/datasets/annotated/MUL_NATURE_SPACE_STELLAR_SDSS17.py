from tabstar_paper.datasets.curation_objects import CuratedTarget
from tabstar_paper.datasets.objects import SupervisedTask

'''
Dataset Name: SDSS17
====
Examples: 78053
====
URL: https://www.openml.org/search?type=data&id=46955
====
Description: This dataset was curated for [TabArena](https://tabarena.ai/) by the TabArena team
as part of the [TabArena Tabular ML IID Study](https://tabarena.ai/data-tabular-ml-iid-study).
For more details on the study, see our [paper](https://tabarena.ai/paper-tabular-ml-iid-study).

**Dataset Focus**: This dataset shall be used for evaluating predictive machine
learning models for independent and identically distributed tabular data. The
intended task is classification.

---
#### Dataset Metadata
- **Licence:** Public Domain
- **Original Data Source:** https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17
- **Reference (please cite)**: fedesoriano. (January 2022). 'Stellar Classification Dataset - SDSS17.' from https://www.kaggle.com/fedesoriano/stellar-classification-dataset-sdss17.
- **Dataset Year:** 2022
- **Dataset Description:** see the reference and the original data source for details.

#### Curation comments by the TabArena team (for code see the [page of the study](https://tabarena.ai/data-tabular-ml-iid-study)):
- We renamed the target feature.
- We dropped duplicates based on "obj_ID" to avoid target leakage from subgroups. 
- We dropped several (ID-like) meta-features that seem to be not part of the predictive task.
====
Target Variable: ObjectType (category, 3 distinct): ['GALAXY', 'STAR', 'QSO']
====
Features:

alpha (float64, 78053 distinct): ['137.2328', '325.7508', '215.0286', '228.4626', '267.6568', '201.7972', '172.3303', '134.7466', '353.5734', '124.1387']
delta (float64, 78053 distinct): ['57.8403', '29.1664', '30.1765', '40.7954', '25.7719', '-1.758', '41.8094', '13.1536', '30.4893', '10.2554']
u (float64, 74160 distinct): ['24.6347', '24.6347', '24.6347', '24.6346', '24.6347', '24.6346', '20.2601', '22.5', '20.9171', '21.7184']
g (float64, 73479 distinct): ['25.1144', '25.1144', '25.1144', '20.4269', '21.6514', '22.4501', '21.9479', '21.7641', '21.3085', '20.464']
r (float64, 72978 distinct): ['24.802', '24.802', '24.802', '19.6175', '20.7653', '20.2045', '21.4462', '20.467', '20.5484', '20.8977']
i (float64, 73029 distinct): ['24.3618', '24.3618', '20.2933', '19.5214', '19.6762', '18.6305', '19.7357', '17.6259', '19.8325', '19.7921']
z (float64, 72933 distinct): ['22.8269', '22.8269', '22.8269', '22.8269', '19.3835', '19.5089', '18.8939', '19.066', '19.9115', '19.0542']
cam_col (uint8, 6 distinct): ['4', '3', '5', '2', '1', '6']
redshift (float64, 77545 distinct): ['0.0', '7.0112', '-0.0041', '0.6291', '0.0042', '0.0042', '0.5714', '2.1128', '0.487', '0.6104']
plate (category, 6264 distinct): ['7147', '7450', '4095', '5061', '6301', '11317', '3656', '7150', '5466', '6516']
fiber_ID (category, 1000 distinct): ['597', '637', '599', '105', '475', '333', '189', '621', '391', '57']
'''

TARGET = CuratedTarget(raw_name="ObjectType", task_type=SupervisedTask.MULTICLASS)
