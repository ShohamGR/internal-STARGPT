from tabstar_paper.datasets.curation_objects import CuratedTarget
from tabstar_paper.datasets.objects import SupervisedTask

'''
Dataset Name: Bank_Customer_Churn
====
Examples: 10000
====
URL: https://www.openml.org/search?type=data&id=46911
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
- **Original Data Source:** https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset
- **Reference (please cite)**: Topre, Gaurav. 'Bank Customer Churn Dataset.' Kaggle, 2022, https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset.
- **Dataset Year:** 2020
- **Dataset Description:** see the reference and the original data source for details.

#### Curation comments by the TabArena team (for code see the [page of the study](https://tabarena.ai/data-tabular-ml-iid-study)):
- We renamed the values of the target variable to be more meaningful.
- We dropped the ID column.

Description Kaggle's shrutime dataset: 45062: 
This data set contains details of a bank's customers and the target variable is a binary variable reflecting the fact whether the customer left the bank (closed his account) or he continues to be a customer.Source: https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling


# Description


About Dataset
This dataset is for ABC Multistate bank with following columns:

customer_id, unused variable.
credit_score, used as input.
country, used as input.
gender, used as input.
age, used as input.
tenure, used as input.
balance, used as input.
products_number, used as input.
credit_card, used as input.
active_member, used as input.
estimated_salary, used as input.
churn, used as the target. 1 if the client has left the bank during some period or 0 if he/she has not.
Aim is to Predict the Customer Churn for ABC Bank.

====
Target Variable: churn (category, 2 distinct): ['No', 'Yes']
====
Features:

credit_score (int64, 460 distinct): ['850', '678', '655', '667', '705', '684', '651', '670', '660', '683']
country (category, 3 distinct): ['France', 'Germany', 'Spain']
gender (category, 2 distinct): ['Male', 'Female']
age (uint8, 70 distinct): ['37', '38', '35', '36', '34', '33', '40', '39', '32', '31']
tenure (uint8, 11 distinct): ['2', '1', '7', '8', '5', '3', '4', '9', '6', '10']
balance (float64, 6382 distinct): ['0.0', '105473.74', '130170.82', '193858.2', '136855.94', '135795.63', '112713.34', '88963.31', '119787.76', '121535.18']
products_number (uint8, 4 distinct): ['1', '2', '3', '4']
credit_card (category, 2 distinct): ['1', '0']
active_member (category, 2 distinct): ['1', '0']
estimated_salary (float64, 9999 distinct): ['24924.92', '41788.37', '172665.21', '87609.5', '9903.42', '64831.36', '40313.47', '164248.33', '189839.93', '159475.08']
'''

CONTEXT = "Bank Customers Churn Prediction Shrutime"
TARGET = CuratedTarget(raw_name="churn", task_type=SupervisedTask.BINARY)