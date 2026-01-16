from tabstar_paper.datasets.curation_objects import CuratedTarget
from tabstar_paper.datasets.objects import SupervisedTask

'''
Dataset Name: Is-this-a-good-customer
====
Examples: 1723
====
URL: https://www.openml.org/search?type=data&id=46938
====
Description: This dataset was curated for [TabArena](https://tabarena.ai/) by the TabArena team
as part of the [TabArena Tabular ML IID Study](https://tabarena.ai/data-tabular-ml-iid-study).
For more details on the study, see our [paper](https://tabarena.ai/paper-tabular-ml-iid-study).

**Dataset Focus**: This dataset shall be used for evaluating predictive machine
learning models for independent and identically distributed tabular data. The
intended task is classification.

---
#### Dataset Metadata
- **Licence:** CC0: Public Domain
- **Original Data Source:** https://www.kaggle.com/datasets/podsyp/is-this-a-good-customer
- **Reference (please cite)**: Kaggle User Podsyp. 'Is This a Good Customer?' Kaggle, 2020, https://www.kaggle.com/datasets/podsyp/is-this-a-good-customer.
- **Dataset Year:** 2020
- **Dataset Description:** see the reference and the original data source for details.

#### Curation comments by the TabArena team (for code see the [page of the study](https://tabarena.ai/data-tabular-ml-iid-study)):
- We renamed the values of the target variable to be more descriptive.

## Kaggle Description

About Dataset
Context
Imbalanced classes put “accuracy” out of business. This is a surprisingly common problem in machine learning (specifically in classification), occurring in datasets with a disproportionate ratio of observations in each class.

Content
Standard accuracy no longer reliably measures performance, which makes model training much trickier.
Imbalanced classes appear in many domains, including:

Antifraud
Antispam
…
Inspiration
5 tactics for handling imbalanced classes in machine learning:

Up-sample the minority class
Down-sample the majority class
Change your performance metric
Penalize algorithms (cost-sensitive training)
Use tree-based algorithms

====
Target Variable: bad_client_target (category, 2 distinct): ['No', 'Yes']
====
Features:

month (uint8, 12 distinct): ['11', '12', '10', '3', '7', '8', '1', '2', '9', '4']
credit_amount (int64, 205 distinct): ['15000', '14000', '11000', '21000', '8000', '13000', '9500', '14500', '18000', '30000']
credit_term (uint8, 22 distinct): ['12', '6', '10', '18', '24', '3', '4', '15', '36', '8']
age (uint8, 66 distinct): ['23', '24', '26', '31', '22', '25', '30', '29', '21', '27']
sex (category, 2 distinct): ['male', 'female']
education (category, 6 distinct): ['Secondary special education', 'Higher education', 'Secondary education', 'Incomplete higher education', 'Incomplete secondary education', 'PhD degree']
product_type (category, 22 distinct): ['Cell phones', 'Household appliances', 'Computers', 'Furniture', 'Clothing', 'Cosmetics and beauty services', 'Windows & Doors', 'Tourism', 'Jewelry', 'Construction Materials']
having_children_flg (category, 2 distinct): ['0', '1']
region (category, 3 distinct): ['2', '0', '1']
income (int64, 76 distinct): ['26000', '31000', '21000', '36000', '16000', '41000', '51000', '19000', '46000', '61000']
family_status (category, 3 distinct): ['Another', 'Married', 'Unmarried']
phone_operator (category, 5 distinct): ['1', '0', '2', '3', '4']
is_client (category, 2 distinct): ['1', '0']
'''

CONTEXT = "Loan Customer Quality Prediction"
TARGET = CuratedTarget(raw_name="bad_client_target", task_type=SupervisedTask.BINARY)
