from tabstar_paper.datasets.curation_objects import CuratedTarget, CuratedFeature
from tabstar_paper.datasets.objects import SupervisedTask

'''
Dataset Name: credit_card_clients_default
====
Examples: 30000
====
URL: https://www.openml.org/search?type=data&id=46919
====
Description: This dataset was curated for [TabArena](https://tabarena.ai/) by the TabArena team
as part of the [TabArena Tabular ML IID Study](https://tabarena.ai/data-tabular-ml-iid-study).
For more details on the study, see our [paper](https://tabarena.ai/paper-tabular-ml-iid-study).

**Dataset Focus**: This dataset shall be used for evaluating predictive machine
learning models for independent and identically distributed tabular data. The
intended task is classification.

# Dataset 43435

Description: Dataset Information
This dataset contains information on default payments, demographic factors, credit data, history of payment, and bill statements of credit card clients in Taiwan from April 2005 to September 2005. 
Content
There are 25 variables:

ID: ID of each client
LIMIT_BAL: Amount of given credit in NT dollars (includes individual and family/supplementary credit
SEX: Gender (1=male, 2=female)
EDUCATION: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
MARRIAGE: Marital status (1=married, 2=single, 3=others)
AGE: Age in years
PAY_0: Repayment status in September, 2005 (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months,  8=payment delay for eight months, 9=payment delay for nine months and above)
PAY_2: Repayment status in August, 2005 (scale same as above)
PAY_3: Repayment status in July, 2005 (scale same as above)
PAY_4: Repayment status in June, 2005 (scale same as above)
PAY_5: Repayment status in May, 2005 (scale same as above)
PAY_6: Repayment status in April, 2005 (scale same as above)
BILL_AMT1: Amount of bill statement in September, 2005 (NT dollar)
BILL_AMT2: Amount of bill statement in August, 2005 (NT dollar)
BILL_AMT3: Amount of bill statement in July, 2005 (NT dollar)
BILL_AMT4: Amount of bill statement in June, 2005 (NT dollar)
BILL_AMT5: Amount of bill statement in May, 2005 (NT dollar)
BILL_AMT6: Amount of bill statement in April, 2005 (NT dollar)
PAY_AMT1: Amount of previous payment in September, 2005 (NT dollar)
PAY_AMT2: Amount of previous payment in August, 2005 (NT dollar)
PAY_AMT3: Amount of previous payment in July, 2005 (NT dollar)
PAY_AMT4: Amount of previous payment in June, 2005 (NT dollar)
PAY_AMT5: Amount of previous payment in May, 2005 (NT dollar)
PAY_AMT6: Amount of previous payment in April, 2005 (NT dollar)
default.payment.next.month: Default payment (1=yes, 0=no)

Inspiration
Some ideas for exploration:

How does the probability of default payment vary by categories of different demographic variables?
Which variables are the strongest predictors of default payment?

Acknowledgements
Any publications based on this dataset should acknowledge the following: 
Lichman, M. (2013). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.
The original dataset can be found here at the UCI Machine Learning Repository.

---
#### Dataset Metadata
- **Licence:** CC BY 4.0
- **Original Data Source:** https://doi.org/10.24432/C55S3H
- **Reference (please cite)**: Yeh, I-Cheng, and Che-hui Lien. 'The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients.' Expert systems with applications 36.2 (2009): 2473-2480. https://doi.org/10.1016/j.eswa.2007.12.020
- **Dataset Year:** 2009
- **Dataset Description:** see the reference and the original data source for details.

#### Curation comments by the TabArena team (for code see the [page of the study](https://tabarena.ai/data-tabular-ml-iid-study)):
- We rename the target variable and restore the original class names.
- We drop the "ID" column.
- Anomaly: the data has temporal features but the task is time-invariant.
====
Target Variable: DefaultOnPaymentNextMonth (category, 2 distinct): ['No', 'Yes']
====
Features:

LIMIT_BAL (int64, 81 distinct): ['50000', '20000', '30000', '80000', '200000', '150000', '100000', '180000', '360000', '60000']
SEX (category, 2 distinct): ['2', '1']
EDUCATION (category, 7 distinct): ['2', '1', '3', '5', '4', '6', '0']
MARRIAGE (category, 4 distinct): ['2', '1', '3', '0']
AGE (uint8, 56 distinct): ['29', '27', '28', '30', '26', '31', '25', '34', '32', '33']
PAY_0 (int64, 11 distinct): ['0', '-1', '1', '-2', '2', '3', '4', '5', '8', '6']
PAY_2 (int64, 11 distinct): ['0', '-1', '2', '-2', '3', '4', '1', '5', '7', '6']
PAY_3 (int64, 11 distinct): ['0', '-1', '-2', '2', '3', '4', '7', '6', '5', '1']
PAY_4 (int64, 11 distinct): ['0', '-1', '-2', '2', '3', '4', '7', '5', '6', '8']
PAY_5 (int64, 10 distinct): ['0', '-1', '-2', '2', '3', '4', '7', '5', '6', '8']
PAY_6 (int64, 10 distinct): ['0', '-1', '-2', '2', '3', '4', '7', '6', '5', '8']
BILL_AMT1 (int64, 22723 distinct): ['0', '390', '780', '326', '316', '2500', '396', '2400', '416', '500']
BILL_AMT2 (int64, 22346 distinct): ['0', '390', '326', '780', '316', '396', '2500', '2400', '-200', '416']
BILL_AMT3 (int64, 22026 distinct): ['0', '390', '780', '326', '316', '396', '2500', '2400', '416', '200']
BILL_AMT4 (int64, 21548 distinct): ['0', '390', '780', '316', '326', '396', '2400', '150', '2500', '416']
BILL_AMT5 (int64, 21010 distinct): ['0', '390', '780', '316', '326', '150', '396', '2400', '2500', '416']
BILL_AMT6 (int64, 20604 distinct): ['0', '390', '780', '150', '316', '326', '396', '416', '-18', '2400']
PAY_AMT1 (int64, 7943 distinct): ['0', '2000', '3000', '5000', '1500', '4000', '10000', '1000', '2500', '6000']
PAY_AMT2 (int64, 7899 distinct): ['0', '2000', '3000', '5000', '1000', '1500', '4000', '10000', '6000', '2500']
PAY_AMT3 (int64, 7518 distinct): ['0', '2000', '1000', '3000', '5000', '1500', '4000', '10000', '1200', '6000']
PAY_AMT4 (int64, 6937 distinct): ['0', '1000', '2000', '3000', '5000', '1500', '4000', '10000', '2500', '500']
PAY_AMT5 (int64, 6897 distinct): ['0', '1000', '2000', '3000', '5000', '1500', '4000', '10000', '500', '6000']
PAY_AMT6 (int64, 6939 distinct): ['0', '1000', '2000', '3000', '5000', '1500', '4000', '10000', '500', '6000']
'''

CONTEXT = "Taiwan Credit Card Clients during 2005"

TARGET = CuratedTarget(raw_name="DefaultOnPaymentNextMonth", task_type=SupervisedTask.BINARY)
FEATURES = [CuratedFeature(raw_name='SEX', value_mapping={'1': 'Male', '2': 'Female'}),
            CuratedFeature(raw_name='EDUCATION',
                           value_mapping={'1': 'Graduate School',
                                          '2': 'University',
                                          '3': 'High School',
                                          '4': 'Others',
                                          '5': 'Unknown 5',
                                          '6': 'Unknown 6',
                                          '0': 'Unknown 0'}),
            CuratedFeature(raw_name='MARRIAGE',
                           value_mapping={'1': 'Married', '2': 'Single', '3': 'Others', '0': 'Unknown'}),]