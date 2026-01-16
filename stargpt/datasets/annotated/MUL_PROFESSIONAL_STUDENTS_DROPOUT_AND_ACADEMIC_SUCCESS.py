from tabstar_paper.datasets.curation_objects import CuratedTarget
from tabstar_paper.datasets.objects import SupervisedTask

'''
Dataset Name: students_dropout_and_academic_success
====
Examples: 4424
====
URL: https://www.openml.org/search?type=data&id=46960
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
- **Original Data Source:** https://doi.org/10.24432/C5MC89
- **Reference (please cite)**: Martins, Monica V., et al. 'Early prediction of students performance in higher education: A case study.' Trends and Applications in Information Systems and Technologies: Volume 1 9. Springer International Publishing, 2021.  https://doi.org/10.1007/978-3-030-72657-7_16
- **Dataset Year:** 2021
- **Dataset Description:** see the reference and the original data source for details.

#### Curation comments by the TabArena team (for code see the [page of the study](https://tabarena.ai/data-tabular-ml-iid-study)):
- We renamed the target variable and fixed some typos in the feature names.
- We reversed the ordinal encoding of all categorical features.
- We removed whitespaces and special characters from the feature names.
====
Target Variable: AcademicOutcome (category, 3 distinct): ['Graduate', 'Dropout', 'Enrolled']
====
Features:

Marital_status (category, 6 distinct): ['single', 'married', 'divorced', 'facto union', 'legally separated', 'widower']
Application_mode (category, 18 distinct): ['1st phase - general contingent', '2nd phase - general contingent', 'Over 23 years old', 'Change of course', 'Technological specialization diploma holders', 'Holders of other higher courses', '3rd phase - general contingent', 'Transfer', 'Change of institution/course', '1st phase - special contingent (Madeira Island)']
Application_order (uint8, 8 distinct): ['1', '2', '3', '4', '5', '6', '9', '0']
Course (category, 17 distinct): ['Nursing', 'Management', 'Social Service', 'Veterinary Nursing', 'Journalism and Communication', 'Management (evening attendance)', 'Advertising and Marketing Management', 'Tourism', 'Communication Design', 'Animation and Multimedia Design']
Daytimeevening_attendance (category, 2 distinct): ['daytime', 'evening']
Previous_qualification (category, 17 distinct): ['Secondary education', 'Technological specialization course', 'Basic education 3rd cycle (9th/10th/11th year) or equiv.', 'Higher education - degree', 'Other - 11th year of schooling', 'Higher education - degree (1st cycle)', 'Professional higher technical course', "Higher education - bachelor's degree", 'Frequency of higher education', '12th year of schooling - not completed']
Previous_qualification_grade (float64, 101 distinct): ['133.1', '130.0', '140.0', '120.0', '150.0', '125.0', '135.0', '110.0', '131.0', '160.0']
Nationality (category, 21 distinct): ['Portuguese', 'Brazilian', 'Santomean', 'Cape Verdean', 'Spanish', 'Guinean', 'Ukrainian', 'Moldova (Republic of)', 'Italian', 'Russian']
Mothers_qualification (category, 29 distinct): ['Secondary Education - 12th Year of Schooling or Eq.', 'Basic education 1st cycle (4th/5th year) or equiv.', 'Basic Education 3rd Cycle', 'Basic Education 2nd Cycle', 'Higher Education - Degree', 'Unknown', "Higher Education - Bachelor's Degree", "Higher Education - Master's", 'Other - 11th Year of Schooling', 'Higher Education - Doctorate']
Fathers_qualification (category, 34 distinct): ['Basic education 1st cycle', 'Basic Education 3rd Cycle', 'Secondary Education - 12th Year of Schooling or Eq.', 'Basic Education 2nd Cycle', 'Higher Education - Degree', 'Unknown', "Higher Education - Bachelor's Degree", "Higher Education - Master's", 'Other - 11th Year of Schooling', 'Technological specialization course']
Mothers_occupation (category, 32 distinct): ['Unskilled Workers', 'Administrative staff', 'Personal Services, Security and Safety Workers and Sellers', 'Intermediate Level Technicians and Professions', 'Specialists in Intellectual/Scientific Activities', 'Skilled Workers in Industry/Construction/Craftsmen', 'Student', 'Representatives of Legislative/Executive Bodies', 'Farmers and Skilled Workers', 'Other Situation']
Fathers_occupation (category, 46 distinct): ['Unskilled Workers', 'Skilled Workers in Industry/Construction/Craftsmen', 'Personal Services, Security and Safety Workers and Sellers', 'Administrative staff', 'Intermediate Level Technicians and Professions', 'Installation and Machine Operators', 'Armed Forces Professions', 'Farmers and Skilled Workers', 'Specialists in Intellectual/Scientific Activities', 'Representatives of Legislative/Executive Bodies']
Admission_grade (float64, 620 distinct): ['130.0', '140.0', '120.0', '100.0', '150.0', '110.0', '160.0', '128.2', '128.0', '123.0']
Displaced (category, 2 distinct): ['yes', 'no']
Educational_special_needs (category, 2 distinct): ['no', 'yes']
Debtor (category, 2 distinct): ['no', 'yes']
Tuition_fees_up_to_date (category, 2 distinct): ['yes', 'no']
Gender (category, 2 distinct): ['female', 'male']
Scholarship_holder (category, 2 distinct): ['no', 'yes']
Age_at_enrollment (uint8, 46 distinct): ['18', '19', '20', '21', '22', '24', '23', '26', '25', '27']
International (category, 2 distinct): ['no', 'yes']
Curricular_units_1st_sem_credited (uint8, 21 distinct): ['0', '2', '1', '3', '6', '4', '5', '7', '8', '9']
Curricular_units_1st_sem_enrolled (uint8, 23 distinct): ['6', '5', '7', '8', '0', '12', '10', '11', '9', '15']
Curricular_units_1st_sem_evaluations (uint8, 35 distinct): ['8', '7', '6', '9', '0', '10', '11', '12', '5', '13']
Curricular_units_1st_sem_approved (uint8, 23 distinct): ['6', '5', '0', '7', '4', '3', '2', '1', '8', '11']
Curricular_units_1st_sem_grade (float64, 805 distinct): ['0.0', '12.0', '13.0', '11.0', '11.5', '14.0', '12.5', '12.3333', '12.6667', '10.0']
Curricular_units_1st_sem_without_evaluations (uint8, 11 distinct): ['0', '1', '2', '3', '4', '7', '6', '5', '8', '12']
Curricular_units_2nd_sem_credited (uint8, 19 distinct): ['0', '1', '2', '4', '5', '3', '6', '11', '7', '9']
Curricular_units_2nd_sem_enrolled (uint8, 22 distinct): ['6', '5', '8', '7', '0', '11', '9', '10', '12', '13']
Curricular_units_2nd_sem_evaluations (uint8, 30 distinct): ['8', '6', '7', '9', '0', '10', '5', '11', '12', '13']
Curricular_units_2nd_sem_approved (uint8, 20 distinct): ['6', '0', '5', '4', '7', '8', '3', '2', '1', '11']
Curricular_units_2nd_sem_grade (float64, 786 distinct): ['0.0', '12.0', '11.0', '13.0', '11.5', '12.5', '10.0', '14.0', '13.5', '12.6667']
Curricular_units_2nd_sem_without_evaluations (uint8, 10 distinct): ['0', '1', '2', '3', '4', '5', '6', '8', '7', '12']
Unemployment_rate (float64, 10 distinct): ['7.6', '9.4', '10.8', '12.4', '12.7', '11.1', '15.5', '13.9', '8.9', '16.2']
Inflation_rate (float64, 9 distinct): ['1.4', '2.6', '-0.8', '0.5', '3.7', '0.6', '2.8', '-0.3', '0.3']
GDP (float64, 10 distinct): ['0.32', '-3.12', '1.74', '1.79', '-1.7', '2.02', '-4.06', '0.79', '3.51', '-0.92']
'''

TARGET = CuratedTarget(raw_name="AcademicOutcome", task_type=SupervisedTask.MULTICLASS)
