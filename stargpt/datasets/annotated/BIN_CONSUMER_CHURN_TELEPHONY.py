from tabstar_paper.datasets.curation_objects import CuratedFeature, CuratedTarget
from tabstar_paper.datasets.objects import SupervisedTask

'''
Dataset Name: churn
====
Examples: 5000
====
URL: https://www.openml.org/search?type=data&id=46915
====
Description: This dataset was curated for [TabArena](https://tabarena.ai/) by the TabArena team
as part of the [TabArena Tabular ML IID Study](https://tabarena.ai/data-tabular-ml-iid-study).
For more details on the study, see our [paper](https://tabarena.ai/paper-tabular-ml-iid-study).

**Dataset Focus**: This dataset shall be used for evaluating predictive machine
learning models for independent and identically distributed tabular data. The
intended task is classification.

---
#### Dataset Metadata
- **Licence:** MIT License
- **Original Data Source:** Original source lost. See instead PMLB (https://github.com/EpistasisLab/pmlb/tree/master/datasets/churn) or OpenML dataset ID 40701.
- **Reference (please cite)**: Marcoulides, George A. 'Discovering Knowledge in Data: an Introduction to Data Mining.' (2005): 1465-1465. https://www.tandfonline.com/doi/abs/10.1198/jasa.2005.s61
- **Dataset Year:** 2005
- **Dataset Description:** see the reference and the original data source for details.

#### Curation comments by the TabArena team (for code see the [page of the study](https://tabarena.ai/data-tabular-ml-iid-study)):
- We renamed the target variable to "CustomerChurned".
- We dropped the "phone_number" feature as it seems to be an index in the original data.
====

# This is the description of dataset: 40701

Description: **Author**: Unknown  
**Source**: [PMLB](https://github.com/EpistasisLab/penn-ml-benchmarks/tree/master/datasets/classification), [BigML](https://bigml.com/user/francisco/gallery/dataset/5163ad540c0b5e5b22000383), Supposedly from UCI but I can't find it there.  
**Please cite**:   

A dataset relating characteristics of telephony account features and usage and whether or not the customer churned. Originally used in [Discovering Knowledge in Data: An Introduction to Data Mining](http://secs.ac.in/wp-content/CSE_PORTAL/DataMining_Daniel.pdf).

====
Target Variable: CustomerChurned (category, 2 distinct): ['No', 'Yes']
====
Features:

state (category, 51 distinct): ['49', '23', '1', '13', '45', '43', '35', '50', '37', '34']
account_length (uint8, 218 distinct): ['90', '87', '105', '93', '112', '100', '86', '101', '116', '103']
area_code (category, 3 distinct): ['415.0', '408.0', '510.0']
international_plan (category, 2 distinct): ['0', '1']
voice_mail_plan (category, 2 distinct): ['0', '1']
number_vmail_messages (uint8, 48 distinct): ['0', '31', '29', '28', '33', '27', '24', '26', '30', '32']
total_day_minutes (float64, 1961 distinct): ['154.0', '189.3', '180.0', '159.5', '177.1', '174.5', '184.5', '183.4', '168.6', '189.8']
total_day_calls (uint8, 123 distinct): ['105', '102', '95', '94', '97', '100', '112', '110', '108', '92']
total_day_charge (float64, 1961 distinct): ['26.18', '32.18', '30.6', '27.12', '30.11', '29.67', '31.37', '31.18', '28.66', '32.27']
total_eve_minutes (float64, 1879 distinct): ['169.9', '199.7', '230.9', '194.0', '210.6', '187.5', '223.5', '167.6', '161.7', '188.8']
total_eve_calls (uint8, 126 distinct): ['105', '97', '91', '103', '94', '101', '104', '96', '102', '109']
total_eve_charge (float64, 1659 distinct): ['15.9', '14.25', '16.12', '16.97', '18.79', '18.96', '19.41', '18.62', '16.35', '16.41']
total_night_minutes (float64, 1853 distinct): ['186.2', '188.2', '194.3', '214.6', '208.9', '228.1', '193.6', '169.4', '192.7', '197.4']
total_night_calls (uint8, 131 distinct): ['105', '102', '100', '104', '99', '103', '91', '94', '98', '95']
total_night_charge (float64, 1028 distinct): ['8.47', '9.66', '8.15', '10.8', '10.26', '9.63', '9.4', '10.49', '9.45', '9.65']
total_intl_minutes (float64, 170 distinct): ['11.1', '9.8', '11.3', '11.4', '10.1', '10.9', '9.7', '10.6', '10.0', '10.5']
total_intl_calls (uint8, 21 distinct): ['3', '4', '2', '5', '6', '7', '1', '8', '9', '10']
total_intl_charge (float64, 170 distinct): ['3.0', '2.65', '3.05', '3.08', '2.73', '2.94', '2.62', '2.86', '2.7', '2.84']
number_customer_service_calls (uint8, 10 distinct): ['1', '2', '0', '3', '4', '5', '6', '7', '9', '8']
'''

CONTEXT = "Telephone Company Customer Churn Prediction"
TARGET = CuratedTarget(raw_name="CustomerChurned", task_type=SupervisedTask.BINARY)
