from tabstar_paper.datasets.curation_objects import CuratedTarget
from tabstar_paper.datasets.objects import SupervisedTask

'''
Dataset Name: E-CommereShippingData
====
Examples: 10999
====
URL: https://www.openml.org/search?type=data&id=46924
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
- **Original Data Source:** https://www.kaggle.com/datasets/prachi13/customer-analytics
- **Reference (please cite)**: Prachi Gopalani. 'E-Commerce Shipping Data.' Kaggle, 2021, https://www.kaggle.com/datasets/prachi13/customer-analytics.
- **Dataset Year:** 2021
- **Dataset Description:** see the reference and the original data source for details.

#### Curation comments by the TabArena team (for code see the [page of the study](https://tabarena.ai/data-tabular-ml-iid-study)):
- We dropped the ID column.
- We renamed the target feature and its values to be more meaningful.
- Anomaly: the target and task seems somewhat disconnected from the features. Moreover, some source information on the data is missing and there might be some translation issues.
- Anomaly: there might be some data issues related to "Warehouse_block" and the value "F" consisting of two block "E" and "F".
====
Target Variable: ArrivedLate (category, 2 distinct): ['Yes', 'No']
====
Features:

Warehouse_block (category, 5 distinct): ['F', 'D', 'A', 'B', 'C']
Mode_of_Shipment (category, 3 distinct): ['Ship', 'Flight', 'Road']
Customer_care_calls (uint8, 6 distinct): ['4', '3', '5', '6', '2', '7']
Customer_rating (uint8, 5 distinct): ['3', '1', '4', '5', '2']
Cost_of_the_Product (int64, 215 distinct): ['245', '257', '260', '254', '243', '264', '255', '263', '258', '266']
Prior_purchases (uint8, 8 distinct): ['3', '2', '4', '5', '6', '10', '7', '8']
Product_importance (category, 3 distinct): ['low', 'medium', 'high']
Gender (category, 2 distinct): ['F', 'M']
Discount_offered (uint8, 65 distinct): ['10', '2', '6', '9', '3', '7', '4', '1', '5', '8']
Weight_in_gms (int64, 4034 distinct): ['4883', '1145', '4314', '4741', '5672', '1005', '5783', '4410', '4562', '1150']
'''

CONTEXT = "E-Commerce Shipping Delay"
TARGET = CuratedTarget(raw_name="ArrivedLate", task_type=SupervisedTask.BINARY)
