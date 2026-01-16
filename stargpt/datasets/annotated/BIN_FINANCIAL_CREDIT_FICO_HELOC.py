from tabstar_paper.datasets.curation_objects import CuratedFeature, CuratedTarget
from tabstar_paper.datasets.objects import SupervisedTask

'''
Dataset Name: heloc
====
Examples: 10459
====
URL: https://www.openml.org/search?type=data&id=46932
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
- **Original Data Source:** https://www.kaggle.com/datasets/averkiyoliabev/home-equity-line-of-creditheloc
- **Reference (please cite)**: Kaggle User Averkiyoliabev. 'Home Equity Line of Credit (HELOC).' Kaggle, 2021, https://www.kaggle.com/datasets/averkiyoliabev/home-equity-line-of-creditheloc.
- **Dataset Year:** 2021
- **Dataset Description:** see the reference and the original data source for details.

#### Curation comments by the TabArena team (for code see the [page of the study](https://tabarena.ai/data-tabular-ml-iid-study)):
- Anomaly: the dataset has time-related features. However, the task and features are preprocessed to be time-invariant.
====

# This is the description of dataset: 45554

====
Description: This dataset is from the "Explainable Machine Learning Challenge":

> The Explainable Machine Learning Challenge is a collaboration between Google, FICO and academics at Berkeley, Oxford, Imperial, UC Irvine and MIT, to generate new research in the area of algorithmic explainability. Teams will be challenged to create machine learning models with both high accuracy and explainability; they will use a real-world financial dataset provided by FICO. Designers and end users of machine learning algorithms will both benefit from more interpretable and explainable algorithms. Machine learning model designers will benefit from Model explanations, written explanations describing the functioning of a trained model. These might include information about which variables or examples are particularly important, they might explain the logic used by an algorithm, and/or characterize input/output relationships between variables and predictions. We expect teams to tell the story of their model such that these explanations will be qualitatively evaluated by data scientists at FICO.

Further information can be retrieved from the [FICO website](https://community.fico.com/s/explainable-machine-learning-challenge).

**Notes**
* We have obtained the dataset from [Kaggle](https://www.kaggle.com/datasets/averkiyoliabev/home-equity-line-of-creditheloc)
* This is a cleaned version of the Kaggle dataset, in which we have removed all rows that only contained `-9`, a special value according to the FAQ.
* Please request access to the data on the FICO website to obtain the full description of the features.
* In this version we have encoded the special values (-9, -8, -7) as missing values to make the data more amenable to non-tree models.
====
Target Variable: RiskPerformance (category, 2 distinct): ['Bad', 'Good']
====
Features:

ExternalRiskEstimate (int64, 61 distinct): ['-9', '65', '66', '68', '73', '72', '70', '63', '75', '69']
MSinceOldestTradeOpen (int64, 526 distinct): ['-9', '-8', '132', '178', '176', '150', '165', '183', '169', '158']
MSinceMostRecentTradeOpen (int64, 112 distinct): ['2', '3', '4', '5', '1', '6', '-9', '7', '8', '9']
AverageMInFile (int64, 237 distinct): ['-9', '79', '71', '74', '68', '80', '75', '84', '63', '70']
NumSatisfactoryTrades (int64, 74 distinct): ['-9', '18', '15', '16', '22', '19', '13', '14', '21', '20']
NumTrades60Ever2DerogPubRec (int64, 19 distinct): ['0', '1', '2', '-9', '3', '4', '5', '6', '7', '8']
NumTrades90Ever2DerogPubRec (int64, 17 distinct): ['0', '1', '-9', '2', '3', '4', '5', '6', '7', '9']
PercentTradesNeverDelq (int64, 72 distinct): ['100', '-9', '96', '97', '95', '94', '93', '92', '88', '90']
MSinceMostRecentDelq (int64, 87 distinct): ['-7', '-9', '1', '2', '3', '4', '5', '-8', '6', '8']
MaxDelq2PublicRecLast12M (int64, 10 distinct): ['7', '6', '4', '-9', '0', '5', '3', '1', '2', '9']
MaxDelqEver (int64, 8 distinct): ['8', '6', '5', '2', '-9', '4', '3', '7']
NumTotalTrades (int64, 88 distinct): ['-9', '15', '16', '17', '22', '20', '24', '21', '18', '19']
NumTradesOpeninLast12M (int64, 19 distinct): ['1', '0', '2', '3', '4', '-9', '5', '6', '7', '8']
PercentInstallTrades (int64, 96 distinct): ['-9', '33', '50', '29', '25', '38', '20', '36', '40', '27']
MSinceMostRecentInqexcl7days (int64, 28 distinct): ['0', '-7', '1', '-9', '-8', '2', '3', '4', '5', '6']
NumInqLast6M (int64, 27 distinct): ['0', '1', '2', '3', '-9', '4', '5', '6', '7', '8']
NumInqLast6Mexcl7days (int64, 27 distinct): ['0', '1', '2', '3', '-9', '4', '5', '6', '7', '8']
NetFractionRevolvingBurden (int64, 128 distinct): ['0', '-9', '1', '2', '3', '4', '5', '6', '-8', '8']
NetFractionInstallBurden (int64, 139 distinct): ['-8', '-9', '100', '92', '83', '77', '87', '75', '95', '89']
NumRevolvingTradesWBalance (int64, 31 distinct): ['2', '3', '4', '1', '5', '6', '-9', '7', '8', '0']
NumInstallTradesWBalance (int64, 20 distinct): ['2', '1', '3', '4', '-8', '-9', '5', '6', '7', '8']
NumBank2NatlTradesWHighUtilization (int64, 19 distinct): ['0', '1', '2', '3', '-9', '-8', '4', '5', '6', '7']
PercentTradesWBalance (int64, 95 distinct): ['100', '50', '67', '-9', '75', '80', '60', '83', '71', '33']
'''

CONTEXT = "Financial Credit Risk of Home Equity Line (HELOC)"
TARGET = CuratedTarget(raw_name="RiskPerformance", task_type=SupervisedTask.BINARY)
