from tabstar_paper.datasets.curation_objects import CuratedTarget
from tabstar_paper.datasets.objects import SupervisedTask

'''
Dataset Name: QSAR_fish_toxicity
====
Examples: 907
====
URL: https://www.openml.org/search?type=data&id=46954
====
Description: This dataset was curated for [TabArena](https://tabarena.ai/) by the TabArena team
as part of the [TabArena Tabular ML IID Study](https://tabarena.ai/data-tabular-ml-iid-study).
For more details on the study, see our [paper](https://tabarena.ai/paper-tabular-ml-iid-study).

**Dataset Focus**: This dataset shall be used for evaluating predictive machine
learning models for independent and identically distributed tabular data. The
intended task is regression.

---
#### Dataset Metadata
- **Licence:** CC BY 4.0
- **Original Data Source:** https://doi.org/10.24432/C5JG7B
- **Reference (please cite)**: Cassotti, Matteo, et al. 'A similarity-based QSAR model for predicting acute toxicity towards the fathead minnow (Pimephales promelas).' SAR and QSAR in Environmental Research 26.3 (2015): 217-243. https://doi.org/10.1080/1062936X.2015.1018938
- **Dataset Year:** 2015
- **Dataset Description:** see the reference and the original data source for details.

#### Curation comments by the TabArena team (for code see the [page of the study](https://tabarena.ai/data-tabular-ml-iid-study)):
- Anomaly: the data contains a lot of duplicates (15%) when ignoring the target feature.
====

# This is the description of dataset: 44970

Description: **Data Description**

Data set containing values for 6 attributes (molecular descriptors) of 908 chemicals used to predict quantitative acute aquatic toxicity towards the fish Pimephales promelas (fathead minnow).

This dataset was used to develop quantitative regression QSAR models to predict acute aquatic toxicity towards the fish Pimephales promelas (fathead minnow) on a set of 908 chemicals. LC50 data, which is the concentration that causes death in 50% of test fish over a test duration of 96 hours, was used as model response.

**Attribute Description**

The model comprised 6 molecular descriptors

1. *CIC0* - information indices
2. *SM1_Dz* - 2D matrix-based descriptors
3. *GATS1i* - 2D autocorrelations
4. *NdsCH* - atom-type counts
5. *NdssC* - atom-type counts
6. *MLOGP* - molecular properties
7. *LC50* - quantitative response, LC50 [-LOG(mol/L)], target feature
====
Target Variable: LC50 (float64, 827 distinct): ['4.208', '3.513', '3.47', '3.926', '3.979', '3.66', '4.416', '4.739', '5.052', '3.112']
====
Features:

CIC0 (float64, 502 distinct): ['2.126', '3.08', '2.377', '2.479', '2.834', '2.508', '2.08', '3.252', '3.012', '2.216']
SM1_Dz(Z) (float64, 186 distinct): ['0.223', '0.134', '0.405', '0.331', '0.0', '0.693', '0.56', '0.496', '0.251', '0.83']
GATS1i (float64, 556 distinct): ['0.941', '1.179', '0.871', '0.938', '0.954', '1.6', '1.189', '1.571', '1.077', '0.95']
NdsCH (uint8, 5 distinct): ['0', '1', '2', '4', '3']
NdssC (uint8, 7 distinct): ['0', '1', '2', '3', '4', '6', '5']
MLOGP (float64, 558 distinct): ['0.8', '1.701', '1.748', '2.604', '1.064', '0.202', '1.587', '2.193', '1.442', '3.291']
'''

CONTEXT = "Fish Toxicity Prediction"
TARGET = CuratedTarget(raw_name="LC50", task_type=SupervisedTask.REGRESSION)