from tabstar_paper.datasets.curation_objects import CuratedTarget
from tabstar_paper.datasets.objects import SupervisedTask

'''
Dataset Name: Another-Dataset-on-used-Fiat-500
====
Examples: 1538
====
URL: https://www.openml.org/search?type=data&id=46907
====
Description: This dataset was curated for [TabArena](https://tabarena.ai/) by the TabArena team
as part of the [TabArena Tabular ML IID Study](https://tabarena.ai/data-tabular-ml-iid-study).
For more details on the study, see our [paper](https://tabarena.ai/paper-tabular-ml-iid-study).

**Dataset Focus**: This dataset shall be used for evaluating predictive machine
learning models for independent and identically distributed tabular data. The
intended task is regression.

---
#### Dataset Metadata
- **Licence:** CC0: Public Domain
- **Original Data Source:** https://www.kaggle.com/datasets/paolocons/another-fiat-500-dataset-1538-rows
- **Reference (please cite)**: Kaggle User Paolocons. 'Another Dataset on Used Fiat 500 (1538 Rows).' 2020. Kaggle, https://www.kaggle.com/datasets/paolocons/another-fiat-500-dataset-1538-rows.
- **Dataset Year:** 2020
- **Dataset Description:** see the reference and the original data source for details.

#### Curation comments by the TabArena team (for code see the [page of the study](https://tabarena.ai/data-tabular-ml-iid-study)):
- N/A
====
Target Variable: price (int64, 222 distinct): ['10500', '10900', '8900', '9900', '9400', '10800', '9500', '7900', '9800', '6900']
====
Features:

model (category, 3 distinct): ['lounge', 'pop', 'sport']
engine_power (uint8, 8 distinct): ['51', '62', '73', '74', '77', '58', '66', '63']
age_in_days (int64, 140 distinct): ['790', '366', '701', '397', '670', '762', '456', '731', '425', '1066']
km (int64, 988 distinct): ['17000', '56779', '19000', '15000', '120000', '21000', '100000', '60000', '18000', '80000']
previous_owners (uint8, 4 distinct): ['1', '2', '3', '4']
lat (float64, 449 distinct): ['41.9032', '41.1079', '45.0697', '45.468', '45.4381', '43.7824', '45.5126', '38.1221', '45.5366', '44.5088']
lon (float64, 450 distinct): ['12.4957', '14.2088', '7.7049', '9.1818', '12.3181', '11.255', '10.329', '13.3611', '10.232', '11.4691']
'''

CONTEXT = "Fiat 500 Used Car Prices"
TARGET = CuratedTarget(raw_name="price", task_type=SupervisedTask.REGRESSION)
