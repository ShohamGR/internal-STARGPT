from tabstar_paper.datasets.curation_objects import CuratedTarget
from tabstar_paper.datasets.objects import SupervisedTask

'''
Dataset Name: healthcare_insurance_expenses
====
Examples: 1338
====
URL: https://www.openml.org/search?type=data&id=46931
====
Description: This dataset was curated for [TabArena](https://tabarena.ai/) by the TabArena team
as part of the [TabArena Tabular ML IID Study](https://tabarena.ai/data-tabular-ml-iid-study).
For more details on the study, see our [paper](https://tabarena.ai/paper-tabular-ml-iid-study).

**Dataset Focus**: This dataset shall be used for evaluating predictive machine
learning models for independent and identically distributed tabular data. The
intended task is regression.

====
Description Dataset 43463: "Insurance-Premium-Data"
This Dataset is something I found online when I wanted to practice regression models. It is an openly available online dataset at multiple places. Though I do not know the exact origin and collection methodology of the data, I would recommend this dataset to everybody who is just beginning their journey in Data science.
====

---
#### Dataset Metadata
- **Licence:** Database Contents License (DbCL) v1.0
- **Original Data Source:** https://www.kaggle.com/datasets/arunjangir245/healthcare-insurance-expenses/
- **Reference (please cite)**: Kaggle User Arunjangir245. 'Healthcare Insurance Expenses.' Kaggle, 2023, https://www.kaggle.com/datasets/arunjangir245/healthcare-insurance-expenses/.
- **Dataset Year:** 2023
- **Dataset Description:** see the reference and the original data source for details.

#### Curation comments by the TabArena team (for code see the [page of the study](https://tabarena.ai/data-tabular-ml-iid-study)):
- N/A
====
Target Variable: charges (float64, 1337 distinct): ['1639.5631', '9095.0683', '11411.685', '8978.1851', '13063.883', '19749.3834', '6940.9099', '14119.62', '2850.6838', '34806.4677']
====
Features:

age (uint8, 47 distinct): ['18', '19', '45', '47', '20', '48', '52', '50', '51', '46']
sex (category, 2 distinct): ['male', 'female']
bmi (float64, 548 distinct): ['32.3', '28.31', '30.8', '31.35', '30.495', '34.1', '30.875', '28.88', '38.06', '33.33']
children (uint8, 6 distinct): ['0', '1', '2', '3', '4', '5']
smoker (category, 2 distinct): ['no', 'yes']
region (category, 4 distinct): ['southeast', 'northwest', 'southwest', 'northeast']
'''

CONTEXT = "Insurance Premium Data for Healthcare Expenses"
TARGET = CuratedTarget(raw_name="charges", task_type=SupervisedTask.REGRESSION)