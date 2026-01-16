from tabstar_paper.datasets.curation_objects import CuratedTarget
from tabstar_paper.datasets.objects import SupervisedTask

'''
Dataset Name: maternal_health_risk
====
Examples: 1014
====
URL: https://www.openml.org/search?type=data&id=46941
====
Description: This dataset was curated for [TabArena](https://tabarena.ai/) by the TabArena team
as part of the [TabArena Tabular ML IID Study](https://tabarena.ai/data-tabular-ml-iid-study).
For more details on the study, see our [paper](https://tabarena.ai/paper-tabular-ml-iid-study).

**Dataset Focus**: This dataset shall be used for evaluating predictive machine
learning models for independent and identically distributed tabular data. The
intended task is classification.

---
#### Dataset Metadata
- **Licence:** CC BY 4.0
- **Original Data Source:** https://doi.org/10.24432/C5DP5D
- **Reference (please cite)**: Ahmed, Marzia, et al. 'Review and analysis of risk factor of maternal health in remote area using the Internet of Things (IoT).' InECCE2019: Proceedings of the 5th International Conference on Electrical, Control & Computer Engineering, Kuantan, Pahang, Malaysia, 29th July 2019. Springer Singapore, 2020. https://doi.org/10.1007/978-981-15-2317-5_30
- **Dataset Year:** 2020
- **Dataset Description:** see the reference and the original data source for details.

#### Curation comments by the TabArena team (for code see the [page of the study](https://tabarena.ai/data-tabular-ml-iid-study)):
- Anomaly: the data has a lot of duplicates (55%).
====
Target Variable: RiskLevel (category, 3 distinct): ['low risk', 'mid risk', 'high risk']
====
Features:

Age (uint8, 50 distinct): ['23', '19', '17', '15', '35', '32', '25', '22', '50', '29']
SystolicBP (uint8, 19 distinct): ['120', '90', '140', '100', '130', '85', '110', '76', '95', '160']
DiastolicBP (uint8, 16 distinct): ['80', '60', '90', '70', '100', '65', '85', '75', '95', '49']
BS (float64, 29 distinct): ['7.5', '6.9', '6.8', '7.0', '7.9', '15.0', '6.1', '11.0', '7.8', '6.7']
BodyTemp (float64, 8 distinct): ['98.0', '101.0', '102.0', '100.0', '103.0', '99.0', '98.4', '98.6']
HeartRate (uint8, 16 distinct): ['70', '76', '80', '77', '66', '60', '88', '86', '78', '82']
'''

TARGET = CuratedTarget(raw_name="RiskLevel", task_type=SupervisedTask.MULTICLASS)
