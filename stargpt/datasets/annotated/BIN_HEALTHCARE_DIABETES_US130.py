from tabstar_paper.datasets.curation_objects import CuratedFeature, CuratedTarget
from tabstar_paper.datasets.objects import SupervisedTask

'''
Dataset Name: Diabetes130US
====
Examples: 101766
====
URL: https://www.openml.org/search?type=data&id=46922
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
- **Original Data Source:** https://doi.org/10.24432/C5230J
- **Reference (please cite)**: Strack, Beata, et al. 'Impact of HbA1c measurement on hospital readmission rates: analysis of 70,000 clinical database patient records.' BioMed research international 2014.1 (2014): 781670. https://doi.org/10.1155/2014/781670
- **Dataset Year:** 2014
- **Dataset Description:** see the reference and the original data source for details.

#### Curation comments by the TabArena team (for code see the [page of the study](https://tabarena.ai/data-tabular-ml-iid-study)):
- We drop duplicated patients based on the "patient_nbr" feature to avoid target leakage.
- We reversed the original ordinal encoding for three ID-based features. 
- We created the target from the "readmitted" column following the original task description.
- We dropped "encounter_id" and "patient_nbr", which are both unique identifiers for each row.
- We keep original "?", NULL-codes, and NaN values because they exist in different ways across the columns.
- Anomaly: There is a distribution shift based on the original order. The reason for this might be that the encounters are ordered in some way such that later parts of the data contain different sub-groups than earlier parts. This is also indicate by the fact that the "payer_code" feature is responsible for the shift. This distribution shift vanishes after randomly shuffling the data (as done by default for this and all other datasets used in TabArena).
====

# This is the description of dataset: 4541

====
Description: **Author**: Attila Reiss, Department Augmented Vision, DFKI, Germany, "attila.reiss '@' dfki.de  
**Date**: August 2012.  
**Source**: UCI  
**Please cite**: Beata Strack, Jonathan P. DeShazo, Chris Gennings, Juan L. Olmo, Sebastian Ventura, Krzysztof J. Cios, and John N. Clore, &ldquo; Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records,&rdquo; BioMed Research International, vol. 2014, Article ID 781670, 11 pages, 2014.  

This data has been prepared to analyze factors related to readmission as well as other outcomes pertaining to patients with diabetes.

**Source**  
The data are submitted on behalf of the Center for Clinical and Translational Research, Virginia Commonwealth University, a recipient of NIH CTSA grant UL1 TR00058 and a recipient of the CERNER data. John Clore (jclore '@' vcu.edu), Krzysztof J. Cios (kcios '@' vcu.edu), Jon DeShazo (jpdeshazo '@' vcu.edu), and Beata Strack (strackb '@' vcu.edu). This data is a de-identified abstract of the Health Facts database (Cerner Corporation, Kansas City, MO).

**Data Set Information**  
The dataset represents 10 years (1999-2008) of clinical care at 130 US hospitals and integrated delivery networks. It includes over 50 features representing patient and hospital outcomes. Information was extracted from the database for encounters that satisfied the following criteria:  
(1) It is an inpatient encounter (a hospital admission).  
(2) It is a diabetic encounter, that is, one during which any kind of diabetes was entered to the system as a diagnosis.  
(3) The length of stay was at least 1 day and at most 14 days.  
(4) Laboratory tests were performed during the encounter.  
(5) Medications were administered during the encounter.  
The data contains such attributes as patient number, race, gender, age, admission type, time in hospital, medical specialty of admitting physician, number of lab test performed, HbA1c test result, diagnosis, number of medication, diabetic medications, number of outpatient, inpatient, and emergency visits in the year before the hospitalization, etc.

**Attribute Information**  
Detailed description of all the attributes is provided in Table 1 of the paper.  

**Relevant Papers**  
Beata Strack, Jonathan P. DeShazo, Chris Gennings, Juan L. Olmo, Sebastian Ventura, Krzysztof J. Cios, and John N. Clore, &ldquo;Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records,&rdquo; BioMed Research International, vol. 2014, Article ID 781670, 11 pages, 2014.

[Web Link](https://www.hindawi.com/journals/bmri/2014/781670/)

Encounter ID	Numeric	Unique identifier of an encounter	0%
Patient number	Numeric	Unique identifier of a patient	0%
Race	Nominal	Values: Caucasian, Asian, African American, Hispanic, and other	2%
Gender	Nominal	Values: male, female, and unknown/invalid	0%
Age	Nominal	Grouped in 10-year intervals: [0, 10), [10, 20), …, [90, 100)	0%
Weight	Numeric	Weight in pounds.	97%
Admission type	Nominal	Integer identifier corresponding to 9 distinct values, for example, emergency, urgent, elective, newborn, and not available	0%
Discharge disposition	Nominal	Integer identifier corresponding to 29 distinct values, for example, discharged to home, expired, and not available	0%
Admission source	Nominal	Integer identifier corresponding to 21 distinct values, for example, physician referral, emergency room, and transfer from a hospital	0%
Time in hospital	Numeric	Integer number of days between admission and discharge	0%
Payer code	Nominal	Integer identifier corresponding to 23 distinct values, for example, Blue Cross/Blue Shield, Medicare, and self-pay	52%
Medical specialty	Nominal	Integer identifier of a specialty of the admitting physician, corresponding to 84 distinct values, for example, cardiology, internal medicine, family/general practice, and surgeon	53%
Number of lab procedures	Numeric	Number of lab tests performed during the encounter	0%
Number of procedures	Numeric	Number of procedures (other than lab tests) performed during the encounter	0%
Number of medications	Numeric	Number of distinct generic names administered during the encounter	0%
Number of outpatient visits	Numeric	Number of outpatient visits of the patient in the year preceding the encounter	0%
Number of emergency visits	Numeric	Number of emergency visits of the patient in the year preceding the encounter	0%
Number of inpatient visits	Numeric	Number of inpatient visits of the patient in the year preceding the encounter	0%
Diagnosis 1	Nominal	The primary diagnosis (coded as first three digits of ICD9); 848 distinct values	0%
Diagnosis 2	Nominal	Secondary diagnosis (coded as first three digits of ICD9); 923 distinct values	0%
Diagnosis 3	Nominal	Additional secondary diagnosis (coded as first three digits of ICD9); 954 distinct values	1%
Number of diagnoses	Numeric	Number of diagnoses entered to the system	0%
Glucose serum test result	Nominal	Indicates the range of the result or if the test was not taken. Values: “>200,” “>300,” “normal,” and “none” if not measured	0%
A1c test result	Nominal	Indicates the range of the result or if the test was not taken. Values: “>8” if the result was greater than 8%, “>7” if the result was greater than 7% but less than 8%, “normal” if the result was less than 7%, and “none” if not measured.	0%
Change of medications	Nominal	Indicates if there was a change in diabetic medications (either dosage or generic name). Values: “change” and “no change”	0%
Diabetes medications	Nominal	Indicates if there was any diabetic medication prescribed. Values: “yes” and “no”	0%
24 features for medications	Nominal	For the generic names: metformin, repaglinide, nateglinide, chlorpropamide, glimepiride, acetohexamide, glipizide, glyburide, tolbutamide, pioglitazone, rosiglitazone, acarbose, miglitol, troglitazone, tolazamide, examide, sitagliptin, insulin, glyburide-metformin, glipizide-metformin, glimepiride-pioglitazone, metformin-rosiglitazone, and metformin-pioglitazone, the feature indicates whether the drug was prescribed or there was a change in the dosage. Values: “up” if the dosage was increased during the encounter, “down” if the dosage was decreased, “steady” if the dosage did not change, and “no” if the drug was not prescribed	0%
Readmitted	Nominal	Days to inpatient readmission. Values: “<30” if the patient was readmitted in less than 30 days, “>30” if the patient was readmitted in more than 30 days, and “No” for no record of readmission.

====
Target Variable: EarlyReadmission (category, 2 distinct): ['No', 'Yes']
====
Features:

race (category, 5 distinct): ['Caucasian', 'AfricanAmerican', 'Hispanic', 'Other', 'Asian']
gender (category, 3 distinct): ['Female', 'Male', 'Unknown/Invalid']
age (category, 10 distinct): ['[70-80)', '[60-70)', '[50-60)', '[80-90)', '[40-50)', '[30-40)', '[90-100)', '[20-30)', '[10-20)', '[0-10)']
weight (category, 9 distinct): ['[75-100)', '[50-75)', '[100-125)', '[125-150)', '[25-50)', '[0-25)', '[150-175)', '[175-200)', '>200']
admission_type_id (category, 8 distinct): ['Emergency', 'Elective', 'Urgent', 'nan', 'Not Available', 'Not Mapped', 'Trauma Center', 'Newborn']
discharge_disposition_id (category, 26 distinct): ['Discharged to home', 'Discharged/transferred to SNF', 'Discharged/transferred to home with home health service', 'nan', 'Discharged/transferred to another short term hospital', 'Discharged/transferred to another rehab fac including rehab units of a hospital .', 'Expired', 'Discharged/transferred to another type of inpatient care institution', 'Not Mapped', 'Discharged/transferred to ICF']
admission_source_id (category, 17 distinct): ['Emergency Room', 'Physician Referral', 'nan', 'Transfer from a hospital', 'Transfer from another health care facility', 'Clinic Referral', 'Transfer from a Skilled Nursing Facility (SNF)', 'Not Mapped', 'HMO Referral', 'Not Available']
time_in_hospital (uint8, 14 distinct): ['3', '2', '1', '4', '5', '6', '7', '8', '9', '10']
payer_code (category, 17 distinct): ['MC', 'HM', 'BC', 'SP', 'MD', 'CP', 'UN', 'CM', 'OG', 'PO']
medical_specialty (category, 70 distinct): ['InternalMedicine', 'Family/GeneralPractice', 'Emergency/Trauma', 'Cardiology', 'Surgery-General', 'Orthopedics', 'Orthopedics-Reconstructive', 'Radiologist', 'Nephrology', 'Pulmonology']
num_lab_procedures (uint8, 116 distinct): ['1', '43', '44', '45', '46', '38', '47', '40', '39', '37']
num_procedures (uint8, 7 distinct): ['0', '1', '2', '3', '6', '4', '5']
num_medications (uint8, 75 distinct): ['13', '12', '11', '15', '14', '10', '16', '9', '17', '8']
number_outpatient (uint8, 33 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
number_emergency (uint8, 18 distinct): ['0', '1', '2', '3', '4', '5', '6', '8', '7', '10']
number_inpatient (uint8, 13 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
diag_1 (category, 696 distinct): ['414', '428', '786', '410', '486', '427', '715', '434', '682', '780']
diag_2 (category, 725 distinct): ['250', '276', '428', '427', '401', '599', '496', '411', '414', '403']
diag_3 (category, 758 distinct): ['250', '401', '276', '428', '427', '414', '496', '272', '403', '599']
number_diagnoses (uint8, 16 distinct): ['9', '5', '6', '7', '8', '4', '3', '2', '1', '16']
max_glu_serum (category, 4 distinct): ['nan', 'Norm', '>200', '>300']
A1Cresult (category, 4 distinct): ['nan', '>8', 'Norm', '>7']
metformin (category, 4 distinct): ['No', 'Steady', 'Up', 'Down']
repaglinide (category, 4 distinct): ['No', 'Steady', 'Up', 'Down']
nateglinide (category, 4 distinct): ['No', 'Steady', 'Up', 'Down']
chlorpropamide (category, 4 distinct): ['No', 'Steady', 'Up', 'Down']
glimepiride (category, 4 distinct): ['No', 'Steady', 'Up', 'Down']
acetohexamide (category, 2 distinct): ['No', 'Steady']
glipizide (category, 4 distinct): ['No', 'Steady', 'Up', 'Down']
glyburide (category, 4 distinct): ['No', 'Steady', 'Up', 'Down']
tolbutamide (category, 2 distinct): ['No', 'Steady']
pioglitazone (category, 4 distinct): ['No', 'Steady', 'Up', 'Down']
rosiglitazone (category, 4 distinct): ['No', 'Steady', 'Up', 'Down']
acarbose (category, 3 distinct): ['No', 'Steady', 'Up']
miglitol (category, 4 distinct): ['No', 'Steady', 'Down', 'Up']
troglitazone (category, 2 distinct): ['No', 'Steady']
tolazamide (category, 2 distinct): ['No', 'Steady']
examide (category, 1 distinct): ['No']
citoglipton (category, 1 distinct): ['No']
insulin (category, 4 distinct): ['No', 'Steady', 'Down', 'Up']
glyburide-metformin (category, 4 distinct): ['No', 'Steady', 'Up', 'Down']
glipizide-metformin (category, 2 distinct): ['No', 'Steady']
glimepiride-pioglitazone (category, 1 distinct): ['No']
metformin-rosiglitazone (category, 2 distinct): ['No', 'Steady']
metformin-pioglitazone (category, 2 distinct): ['No', 'Steady']
change (category, 2 distinct): ['No', 'Ch']
diabetesMed (category, 2 distinct): ['Yes', 'No']
'''

CONTEXT = "Diabetes Patients Readmission Prediction in US Hospitals"
TARGET = CuratedTarget(raw_name="EarlyReadmission", task_type=SupervisedTask.BINARY)