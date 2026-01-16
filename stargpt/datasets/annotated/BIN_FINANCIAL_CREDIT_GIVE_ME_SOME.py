from tabstar_paper.datasets.curation_objects import CuratedTarget
from tabstar_paper.datasets.objects import SupervisedTask

'''
Dataset Name: GiveMeSomeCredit
====
Examples: 150000
====
URL: https://www.openml.org/search?type=data&id=46929
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
- **Original Data Source:** https://www.kaggle.com/competitions/GiveMeSomeCredit
- **Reference (please cite)**: Credit Fusion and Will Cukierski. Give Me Some Credit. https://kaggle.com/competitions/GiveMeSomeCredit, 2011. Kaggle.
- **Dataset Year:** 2011
- **Dataset Description:** see the reference and the original data source for details.

#### Curation comments by the TabArena team (for code see the [page of the study](https://tabarena.ai/data-tabular-ml-iid-study)):
- We renamed the target feature and its value to be more descriptive.
====

# This is the description of dataset: 45577

====
Description: Improve on the state of the art in credit scoring by predicting the probability that somebody will experience financial distress in the next two years.

## Description

Banks play a crucial role in market economies. They decide who can get finance and on what terms and can make or break investment decisions. For markets and society to function, individuals and companies need access to credit. 

Credit scoring algorithms, which make a guess at the probability of default, are the method banks use to determine whether or not a loan should be granted. This competition requires participants to improve on the state of the art in credit scoring, by predicting the probability that somebody will experience financial distress in the next two years.

The goal of this competition is to build a model that borrowers can use to help make the best financial decisions.

Historical data are provided on 250,000 borrowers and the prize pool is $5,000 ($3,000 for first, $1,500 for second and $500 for third).

## Features

| Variable Name                        | Description                                                                                                                                              | Type       |
|--------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|------------|
| SeriousDlqin2yrs                     | Person experienced 90 days past due delinquency or worse                                                                                                 | Y/N        |
| RevolvingUtilizationOfUnsecuredLines | Total balance on credit cards and personal lines of credit except real estate and no installment debt like car loans divided by the sum of credit limits | percentage |
| age                                  | Age of borrower in years                                                                                                                                 | integer    |
| NumberOfTime30-59DaysPastDueNotWorse | Number of times borrower has been 30-59 days past due but no worse in the last 2 years.                                                                  | integer    |
| DebtRatio                            | Monthly debt payments, alimony,living costs divided by monthy gross income                                                                               | percentage |
| MonthlyIncome                        | Monthly income                                                                                                                                           | real       |
| NumberOfOpenCreditLinesAndLoans      | Number of Open loans (installment like car loan or mortgage) and Lines of credit (e.g. credit cards)                                                     | integer    |
| NumberOfTimes90DaysLate              | Number of times borrower has been 90 days or more past due.                                                                                              | integer    |
| NumberRealEstateLoansOrLines         | Number of mortgage and real estate loans including home equity lines of credit                                                                           | integer    |
| NumberOfTime60-89DaysPastDueNotWorse | Number of times borrower has been 60-89 days past due but no worse in the last 2 years.                                                                  | integer    |
| NumberOfDependents                   | Number of dependents in family excluding themselves (spouse, children etc.)                                                                              | integer    |

**Note:** This is the training part of the Kaggle competition going by the dataset name hosted [here](https://www.kaggle.com/competitions/GiveMeSomeCredit/overview).

Target Variable: FinancialDistressNextTwoYears (category, 2 distinct): ['No', 'Yes']
====
Features:

RevolvingUtilizationOfUnsecuredLines (float64, 125728 distinct): ['0.0', '1.0', '1.0', '0.9501', '0.7131', '0.008', '0.9541', '0.7964', '0.988', '0.994']
age (uint8, 86 distinct): ['49', '48', '50', '63', '47', '46', '53', '51', '52', '56']
NumberOfTime30-59DaysPastDueNotWorse (uint8, 16 distinct): ['0', '1', '2', '3', '4', '5', '98', '6', '7', '8']
DebtRatio (float64, 114194 distinct): ['0.0', '1.0', '4.0', '2.0', '3.0', '5.0', '9.0', '10.0', '7.0', '13.0']
MonthlyIncome (float64, 13594 distinct): ['5000.0', '4000.0', '6000.0', '3000.0', '0.0', '2500.0', '10000.0', '3500.0', '4500.0', '7000.0']
NumberOfOpenCreditLinesAndLoans (uint8, 58 distinct): ['6', '7', '5', '8', '4', '9', '10', '3', '11', '12']
NumberOfTimes90DaysLate (uint8, 19 distinct): ['0', '1', '2', '3', '4', '98', '5', '6', '7', '8']
NumberRealEstateLoansOrLines (uint8, 28 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
NumberOfTime60-89DaysPastDueNotWorse (uint8, 13 distinct): ['0', '1', '2', '3', '98', '4', '5', '6', '7', '96']
NumberOfDependents (float64, 13 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
'''

CONTEXT = "Credit Scoring for Financial Distress Prediction"
TARGET = CuratedTarget(raw_name="FinancialDistressNextTwoYears", task_type=SupervisedTask.BINARY)
