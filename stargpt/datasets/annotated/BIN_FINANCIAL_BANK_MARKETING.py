from tabstar_paper.datasets.curation_objects import CuratedFeature, CuratedTarget
from tabstar_paper.datasets.objects import SupervisedTask

'''
Dataset Name: bank-marketing
====
Examples: 45211
====
URL: https://www.openml.org/search?type=data&id=46910
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
- **Original Data Source:** https://doi.org/10.24432/C5K306
- **Reference (please cite)**: Moro, Sergio, Paulo Cortez, and Paulo Rita. 'A data-driven approach to predict the success of bank telemarketing.' Decision Support Systems 62 (2014): 22-31. https://doi.org/10.1016/j.dss.2014.03.001
- **Dataset Year:** 2012
- **Dataset Description:** see the reference and the original data source for details.

#### Curation comments by the TabArena team (for code see the [page of the study](https://tabarena.ai/data-tabular-ml-iid-study)):
- We removed the "duration" feature following its original description to obtain a "realistic predictive model".
- We further remove the "month" and "day_of_week" features, as they also relate to the last contact -- which is not available in a real-world scenario.

====

# This is the description of dataset: 1461

Description: **Author**: Paulo Cortez, Sérgio Moro
**Source**: [UCI](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
**Please cite**: S. Moro, R. Laureano and P. Cortez. Using Data Mining for Bank Direct Marketing: An Application of the CRISP-DM Methodology. In P. Novais et al. (Eds.), Proceedings of the European Simulation and Modelling Conference - ESM'2011, pp. 117-121, Guimarães, Portugal, October, 2011. EUROSIS.       

**Bank Marketing**  
The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be (or not) subscribed. 

The classification goal is to predict if the client will subscribe a term deposit (variable y).

### Attribute information  
For more information, read [Moro et al., 2011].

Input variables:

- bank client data:

1 - age (numeric) 

2 - job : type of job (categorical: "admin.","unknown","unemployed","management","housemaid","entrepreneur", "student","blue-collar","self-employed","retired","technician","services") 

3 - marital : marital status (categorical: "married","divorced","single"; note: "divorced"  means divorced or widowed) 

4 - education (categorical: "unknown","secondary","primary","tertiary") 

5 - default: has credit in default? (binary: "yes","no") 

6 - balance: average yearly balance, in euros (numeric) 

7 - housing: has housing loan? (binary: "yes","no") 

8 - loan: has personal loan? (binary: "yes","no")

- related with the last contact of the current campaign:

9 - contact: contact communication type (categorical: "unknown","telephone","cellular")

10 - day: last contact day of the month (numeric)

11 - month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")

12 - duration: last contact duration, in seconds (numeric)

- other attributes:

13 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)

14 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted) 

15 - previous: number of contacts performed before this campaign and for this client (numeric) 

16 - poutcome: outcome of the previous marketing campaign (categorical: "unknown","other","failure","success")
 
- output variable (desired target):

17 - y - has the client subscribed a term deposit? (binary: "yes","no")
====

====
Target Variable: SubscribeTermDeposit (category, 2 distinct): ['no', 'yes']
====
Features:

age (uint8, 77 distinct): ['32', '31', '33', '34', '35', '36', '30', '37', '39', '38']
job (category, 12 distinct): ['blue-collar', 'management', 'technician', 'admin.', 'services', 'retired', 'self-employed', 'entrepreneur', 'unemployed', 'housemaid']
marital (category, 3 distinct): ['married', 'single', 'divorced']
education (category, 4 distinct): ['secondary', 'tertiary', 'primary', 'unknown']
default (category, 2 distinct): ['no', 'yes']
balance (int64, 7168 distinct): ['0', '1', '2', '4', '3', '5', '6', '8', '23', '7']
housing (category, 2 distinct): ['yes', 'no']
loan (category, 2 distinct): ['no', 'yes']
contact (category, 3 distinct): ['cellular', 'unknown', 'telephone']
campaign (uint8, 48 distinct): ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
pdays (int64, 559 distinct): ['-1', '182', '92', '183', '91', '181', '370', '184', '364', '95']
previous (int64, 41 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
poutcome (object, 4 distinct): ['unknown', 'failure', 'other', 'success']
'''

CONTEXT = "Direct Phone Marketing Campaigns of a Portuguese Banking Institution"
TARGET = CuratedTarget(raw_name="SubscribeTermDeposit", task_type=SupervisedTask.BINARY)
