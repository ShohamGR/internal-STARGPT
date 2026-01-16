from tabstar_paper.datasets.curation_objects import CuratedTarget
from tabstar_paper.datasets.objects import SupervisedTask

'''
Dataset Name: Fitness_Club
====
Examples: 1500
====
URL: https://www.openml.org/search?type=data&id=46927
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
- **Original Data Source:** https://www.kaggle.com/datasets/ddosad/datacamps-data-science-associate-certification
- **Reference (please cite)**: Kaggle User Ddosad. 'Fitness Club Dataset for ML Classification.' Kaggle, 2023, https://www.kaggle.com/datasets/ddosad/datacamps-data-science-associate-certification.
- **Dataset Year:** 2023
- **Dataset Description:** see the reference and the original data source for details.

#### Curation comments by the TabArena team (for code see the [page of the study](https://tabarena.ai/data-tabular-ml-iid-study)):
- We dropped the ID column.
- We renamed the values of the target variable to be more meaningful.
- We treat "-0" values as "0" values for the target, following the description.
- We removed trailing words (like "days") from "days_before".
- We aligned the naming of "day_of_week" to be consistent per day.
====
Target Variable: attended (category, 2 distinct): ['No', 'Yes']
====
Features:

months_as_member (uint8, 72 distinct): ['8', '7', '6', '9', '5', '12', '11', '10', '13', '15']
weight (float64, 1241 distinct): ['78.28', '84.64', '75.63', '71.74', '74.76', '101.27', '81.36', '76.99', '80.54', '74.69']
days_before (uint8, 19 distinct): ['10', '2', '8', '12', '14', '4', '6', '7', '3', '5']
day_of_week (category, 7 distinct): ['Fri', 'Thu', 'Mon', 'Sun', 'Sat', 'Tue', 'Wed']
time (category, 2 distinct): ['AM', 'PM']
category (category, 6 distinct): ['HIIT', 'Cycling', 'Strength', 'Yoga', 'Aqua', '-']
'''

CONTEXT = "Attendance of members to a fitness club."
TARGET = CuratedTarget(raw_name="attended", task_type=SupervisedTask.BINARY)
