from tabstar_paper.datasets.curation_objects import CuratedTarget
from tabstar_paper.datasets.objects import SupervisedTask

'''
Dataset Name: HR_Analytics_Job_Change_of_Data_Scientists
====
Examples: 19158
====
URL: https://www.openml.org/search?type=data&id=46935
====
Description: This dataset was curated for [TabArena](https://tabarena.ai/) by the TabArena team
as part of the [TabArena Tabular ML IID Study](https://tabarena.ai/data-tabular-ml-iid-study).
For more details on the study, see our [paper](https://tabarena.ai/paper-tabular-ml-iid-study).

**Dataset Focus**: This dataset shall be used for evaluating predictive machine
learning models for independent and identically distributed tabular data. The
intended task is classification.

---
#### Dataset Metadata
- **Licence:** CC0: Public Domain
- **Original Data Source:** https://www.kaggle.com/datasets/arashnic/hr-analytics-job-change-of-data-scientists
- **Reference (please cite)**: Kaggle User Arashnic. 'HR Analytics: Job Change of Data Scientists.' Kaggle, 2021, https://www.kaggle.com/datasets/arashnic/hr-analytics-job-change-of-data-scientists.
- **Dataset Year:** 2021
- **Dataset Description:** see the reference and the original data source for details.

#### Curation comments by the TabArena team (for code see the [page of the study](https://tabarena.ai/data-tabular-ml-iid-study)):
- We renamed the target feature and its values to be more descriptive.
- We drop the ID column.
====
Target Variable: LookingForJobChange (category, 2 distinct): ['No', 'Yes']
====
Features:

city (category, 123 distinct): ['city_103', 'city_21', 'city_16', 'city_114', 'city_160', 'city_136', 'city_67', 'city_75', 'city_102', 'city_104']
city_development_index (float64, 93 distinct): ['0.92', '0.624', '0.91', '0.926', '0.698', '0.897', '0.939', '0.855', '0.804', '0.924']
gender (category, 4 distinct): ['Male', 'nan', 'Female', 'Other']
relevent_experience (category, 2 distinct): ['Has relevent experience', 'No relevent experience']
enrolled_university (category, 4 distinct): ['no_enrollment', 'Full time course', 'Part time course', 'nan']
education_level (category, 6 distinct): ['Graduate', 'Masters', 'High School', 'nan', 'Phd', 'Primary School']
major_discipline (category, 7 distinct): ['STEM', 'nan', 'Humanities', 'Other', 'Business Degree', 'Arts', 'No Major']
experience (object, 22 distinct): ['>20', '5', '4', '3', '6', '2', '7', '10', '9', '8']
company_size (category, 9 distinct): ['nan', '50-99', '100-500', '10000+', '10/49', '1000-4999', '<10', '500-999', '5000-9999']
company_type (category, 7 distinct): ['Pvt Ltd', 'nan', 'Funded Startup', 'Public Sector', 'Early Stage Startup', 'NGO', 'Other']
last_new_job (category, 7 distinct): ['1', '>4', '2', 'never', '4', '3', 'nan']
training_hours (int64, 241 distinct): ['28', '12', '18', '22', '50', '20', '17', '24', '6', '34']
'''

TARGET = CuratedTarget(raw_name="LookingForJobChange", task_type=SupervisedTask.BINARY)
