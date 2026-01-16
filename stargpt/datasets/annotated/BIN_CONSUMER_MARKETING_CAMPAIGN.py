from tabstar_paper.datasets.curation_objects import CuratedTarget, CuratedFeature
from tabstar_paper.datasets.objects import SupervisedTask, FeatureType

'''
Dataset Name: Marketing_Campaign
====
Examples: 2240
====
URL: https://www.openml.org/search?type=data&id=46940
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
- **Original Data Source:** https://www.kaggle.com/datasets/rodsaldanha/arketing-campaign
- **Reference (please cite)**: Saldanha, Rodolfo. 'Marketing Campaign.' Kaggle, 2020, https://www.kaggle.com/datasets/rodsaldanha/arketing-campaign.
- **Dataset Year:** 2020
- **Dataset Description:** see the reference and the original data source for details.

#### Curation comments by the TabArena team (for code see the [page of the study](https://tabarena.ai/data-tabular-ml-iid-study)):
- We drop the ID column.
- We rename the values of the target variable to be more descriptive.
- We drop constant columns.

# Kaggle


About Dataset
Context
A response model can provide a significant boost to the efficiency of a marketing campaign by increasing responses or reducing expenses. The objective is to predict who will respond to an offer for a product or service

Content
AcceptedCmp1 - 1 if customer accepted the offer in the 1st campaign, 0 otherwise
AcceptedCmp2 - 1 if customer accepted the offer in the 2nd campaign, 0 otherwise
AcceptedCmp3 - 1 if customer accepted the offer in the 3rd campaign, 0 otherwise
AcceptedCmp4 - 1 if customer accepted the offer in the 4th campaign, 0 otherwise
AcceptedCmp5 - 1 if customer accepted the offer in the 5th campaign, 0 otherwise
Response (target) - 1 if customer accepted the offer in the last campaign, 0 otherwise
Complain - 1 if customer complained in the last 2 years
DtCustomer - date of customer’s enrolment with the company
Education - customer’s level of education
Marital - customer’s marital status
Kidhome - number of small children in customer’s household
 Teenhome - number of teenagers in customer’s household
 Income - customer’s yearly household income
MntFishProducts - amount spent on fish products in the last 2 years
MntMeatProducts - amount spent on meat products in the last 2 years
MntFruits - amount spent on fruits products in the last 2 years
MntSweetProducts - amount spent on sweet products in the last 2 years
MntWines - amount spent on wine products in the last 2 years
MntGoldProds - amount spent on gold products in the last 2 years
NumDealsPurchases - number of purchases made with discount
NumCatalogPurchases - number of purchases made using catalogue
NumStorePurchases - number of purchases made directly in stores
NumWebPurchases - number of purchases made through company’s web site
NumWebVisitsMonth - number of visits to company’s web site in the last month
Recency - number of days since the last purchase

Acknowledgements
O. Parr-Rud. Business Analytics Using SAS Enterprise Guide and SAS Enterprise Miner. SAS Institute, 2014.

Inspiration
The main objective is to train a predictive model which allows the company to maximize the profit of the next marketing campaign.
====
Target Variable: Response (category, 2 distinct): ['No', 'Yes']
====
Features:

Year_Birth (int64, 59 distinct): ['1976', '1971', '1975', '1972', '1978', '1970', '1973', '1965', '1969', '1974']
Education (category, 5 distinct): ['Graduation', 'PhD', 'Master', '2n Cycle', 'Basic']
Marital_Status (category, 8 distinct): ['Married', 'Together', 'Single', 'Divorced', 'Widow', 'Alone', 'Absurd', 'YOLO']
Income (float64, 1974 distinct): ['7500.0', '35860.0', '83844.0', '39922.0', '80134.0', '34176.0', '37760.0', '46098.0', '47025.0', '18929.0']
Kidhome (uint8, 3 distinct): ['0', '1', '2']
Teenhome (uint8, 3 distinct): ['0', '1', '2']
Dt_Customer (object, 663 distinct): ['2012-08-31', '2012-09-12', '2013-02-14', '2014-05-12', '2014-05-22', '2013-08-20', '2014-04-05', '2012-10-29', '2014-03-23', '2014-03-01']
Recency (uint8, 100 distinct): ['56', '30', '54', '46', '49', '92', '65', '3', '71', '29']
MntWines (int64, 776 distinct): ['2', '5', '6', '1', '4', '8', '3', '9', '12', '14']
MntFruits (uint8, 158 distinct): ['0', '1', '2', '3', '4', '7', '5', '6', '12', '8']
MntMeatProducts (int64, 558 distinct): ['7', '5', '11', '8', '6', '10', '3', '9', '16', '12']
MntFishProducts (int64, 182 distinct): ['0', '2', '3', '4', '6', '7', '8', '10', '13', '12']
MntSweetProducts (int64, 177 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '12']
MntGoldProds (int64, 213 distinct): ['1', '4', '3', '5', '12', '2', '0', '6', '7', '10']
NumDealsPurchases (uint8, 15 distinct): ['1', '2', '3', '4', '5', '6', '0', '7', '8', '9']
NumWebPurchases (uint8, 15 distinct): ['2', '1', '3', '4', '5', '6', '7', '8', '9', '0']
NumCatalogPurchases (uint8, 14 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '10']
NumStorePurchases (uint8, 14 distinct): ['3', '4', '2', '5', '6', '8', '7', '10', '9', '12']
NumWebVisitsMonth (uint8, 16 distinct): ['7', '8', '6', '5', '4', '3', '2', '1', '9', '0']
AcceptedCmp3 (category, 2 distinct): ['0', '1']
AcceptedCmp4 (category, 2 distinct): ['0', '1']
AcceptedCmp5 (category, 2 distinct): ['0', '1']
AcceptedCmp1 (category, 2 distinct): ['0', '1']
AcceptedCmp2 (category, 2 distinct): ['0', '1']
Complain (category, 2 distinct): ['0', '1']

'''

CONTEXT = "Consumer Response for Marketing Campaigns"
TARGET = CuratedTarget(raw_name="Response", task_type=SupervisedTask.BINARY)
FEATURES = [CuratedFeature(raw_name="Dt_Customer", feat_type=FeatureType.DATE)]
