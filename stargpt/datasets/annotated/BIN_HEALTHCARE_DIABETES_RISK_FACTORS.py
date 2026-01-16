from tabstar_paper.datasets.curation_objects import CuratedTarget
from tabstar_paper.datasets.objects import SupervisedTask

'''
Dataset Name: diabetes
====
Examples: 768
====
URL: https://www.openml.org/search?type=data&id=46921
====
Description: This dataset was curated for [TabArena](https://tabarena.ai/) by the TabArena team
as part of the [TabArena Tabular ML IID Study](https://tabarena.ai/data-tabular-ml-iid-study).
For more details on the study, see our [paper](https://tabarena.ai/paper-tabular-ml-iid-study).

**Dataset Focus**: This dataset shall be used for evaluating predictive machine
learning models for independent and identically distributed tabular data. The
intended task is classification.

Description for 46254
The dataset, named 'diabetes.csv', serves as a comprehensive resource for understanding various factors that may influence the occurrence of diabetes in individuals. Consisting of several medically relevant parameters, the dataset captures key details across 9 columns, namely Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI (Body Mass Index), DiabetesPedigreeFunction, Age, and Outcome. Each column reflects a distinct attribute significant to diabetes research and potential predictive modeling.

Attribute Description:
1. Pregnancies: Number of times pregnant (Example values: 2, 1)
2. Glucose: Plasma glucose concentration over 2 hours in an oral glucose tolerance test (Example values: 82, 142)
3. BloodPressure: Diastolic blood pressure (mm Hg) (Example values: 70, 64)
4. SkinThickness: Triceps skin fold thickness (mm) (Example values: 27, 0)
5. Insulin: 2-Hour serum insulin (mu U/ml) (Example values: 168, 0)
6. BMI: Body mass index (weight in kg/(height in m)^2) (Example values: 36.8, 30.1)
7. DiabetesPedigreeFunction: Diabetes pedigree function (Example values: 0.34, 0.396)
8. Age: Age in years (Example values: 54, 24)
9. Outcome: Class variable (0 or 1) where 1 denotes the presence of diabetes and 0 denotes absence (Example values: 1, 0)

Use Case:
This dataset is particularly useful for medical researchers, data scientists, and healthcare providers seeking to identify patterns or factors that significantly contribute to diabetes. By employing statistical analysis or machine learning models, one can predict the likelihood of diabetes occurrence based on the dataset's parameters. Furthermore, this dataset can facilitate a better understanding of how various factors, such as pregnancy, BMI, and age, interact with each other in the context of diabetes, thereby aiding in preventative healthcare planning and patient education.

---
#### Dataset Metadata
- **Licence:** CC0: Public Domain
- **Original Data Source:** Original UCI source is lost, backup on [OpenML](https://www.openml.org/search?type=data&sort=runs&id=37) and [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).
- **Reference (please cite)**: Smith, Jack W., et al. 'Using the ADAP learning algorithm to forecast the onset of diabetes mellitus.' Proceedings of the annual symposium on computer application in medical care. 1988. https://pmc.ncbi.nlm.nih.gov/articles/PMC2245318/
- **Dataset Year:** 1988
- **Dataset Description:** see the reference and the original data source for details.

#### Curation comments by the TabArena team (for code see the [page of the study](https://tabarena.ai/data-tabular-ml-iid-study)):
- We renamed the class variable "Outcome" to "TestedPositiveForDiabetes" and replaced 1 with yes, and 0 with no.
====
Target Variable: TestedPositiveForDiabetes (category, 2 distinct): ['No', 'Yes']
====
Features:

Pregnancies (uint8, 17 distinct): ['1', '0', '2', '3', '4', '5', '6', '7', '8', '9']
Glucose (uint8, 136 distinct): ['99', '100', '111', '129', '106', '125', '108', '105', '95', '112']
BloodPressure (uint8, 47 distinct): ['70', '74', '78', '68', '72', '64', '80', '76', '60', '0']
SkinThickness (uint8, 51 distinct): ['0', '32', '30', '27', '23', '33', '18', '28', '31', '39']
Insulin (int64, 186 distinct): ['0', '105', '140', '130', '120', '180', '94', '100', '110', '135']
BMI (float64, 248 distinct): ['32.0', '31.2', '31.6', '0.0', '33.3', '32.4', '30.1', '32.8', '32.9', '30.8']
DiabetesPedigreeFunction (float64, 517 distinct): ['0.254', '0.258', '0.259', '0.207', '0.261', '0.268', '0.238', '0.692', '0.245', '0.27']
Age (uint8, 52 distinct): ['22', '21', '25', '24', '23', '28', '26', '27', '29', '31']
'''

CONTEXT = "Diabetes Risk Factors"
TARGET = CuratedTarget(raw_name="TestedPositiveForDiabetes", task_type=SupervisedTask.BINARY)
