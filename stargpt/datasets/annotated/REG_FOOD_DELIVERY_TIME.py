from tabstar_paper.datasets.curation_objects import CuratedTarget
from tabstar_paper.datasets.objects import SupervisedTask

'''
Dataset Name: Food_Delivery_Time
====
Examples: 45451
====
URL: https://www.openml.org/search?type=data&id=46928
====
Description: This dataset was curated for [TabArena](https://tabarena.ai/) by the TabArena team
as part of the [TabArena Tabular ML IID Study](https://tabarena.ai/data-tabular-ml-iid-study).
For more details on the study, see our [paper](https://tabarena.ai/paper-tabular-ml-iid-study).

**Dataset Focus**: This dataset shall be used for evaluating predictive machine
learning models for independent and identically distributed tabular data. The
intended task is regression.

---
#### Dataset Metadata
- **Licence:** Database Contents License (DbCL) v1.0
- **Original Data Source:** https://www.kaggle.com/datasets/rajatkumar30/food-delivery-time
- **Reference (please cite)**: Kaggle User Rajatkumar30. 'Food Delivery Time.' Kaggle, 2023, https://www.kaggle.com/datasets/rajatkumar30/food-delivery-time.
- **Dataset Year:** 2023
- **Dataset Description:** see the reference and the original data source for details.

#### Curation comments by the TabArena team (for code see the [page of the study](https://tabarena.ai/data-tabular-ml-iid-study)):
- We dropped entries with a duplicated ID, keeping only the first one.
- We dropped the ID column.
- Anomaly: The ID of the delivery person is given as a feature. In some contexts, this feature might not be allowed to use. Moreover, using this information might require considering a temporal split where a cold-start problem is covered.

# Kaggle Description
About Dataset
** Please Upvote if you like the dataset **

Food delivery Dataset
Predict the time taken by the delivery person to deliver the food from the restaurant to the delivery location. With the help of age of the delivery person, previous rating and distance between restaurant and delivery location.
Dataset Columns:-

ID
Delivery_person_ID
Delivery_person_Age
Delivery_person_Ratings
Restaurant_latitude
Restaurant_longitude
Delivery_location_latitude
Delivery_location_longitude
Type_of_order
Type_of_vehicle
** If You Like this Dataset Please UPVOTE!!! **
====
Target Variable: Time_taken(min) (uint8, 45 distinct): ['26', '25', '27', '28', '29', '19', '15', '18', '16', '17']
====
Features:

Delivery_person_ID (category, 1320 distinct): ['JAPRES11DEL02', 'HYDRES04DEL02', 'JAPRES03DEL01', 'PUNERES01DEL01', 'RANCHIRES02DEL01', 'BANGRES03DEL01', 'JAPRES09DEL02', 'SURRES11DEL01', 'VADRES11DEL01', 'INDORES08DEL02']
Delivery_person_Age (uint8, 22 distinct): ['29', '35', '36', '37', '30', '38', '24', '32', '22', '33']
Delivery_person_Ratings (float64, 28 distinct): ['4.6', '4.8', '4.7', '4.9', '5.0', '4.5', '4.1', '4.2', '4.3', '4.4']
Restaurant_latitude (float64, 653 distinct): ['0.0', '26.9114', '26.9141', '26.9029', '26.8923', '26.9029', '26.8884', '26.9053', '26.9137', '26.9135']
Restaurant_longitude (float64, 515 distinct): ['0.0', '75.8057', '75.789', '75.8069', '75.793', '75.7929', '75.8007', '75.7528', '75.7946', '75.8373']
Delivery_location_latitude (float64, 4373 distinct): ['0.13', '0.02', '0.06', '0.09', '0.07', '0.04', '0.05', '0.11', '0.01', '0.08']
Delivery_location_longitude (float64, 4373 distinct): ['0.13', '0.02', '0.06', '0.09', '0.07', '0.04', '0.05', '0.11', '0.01', '0.08']
Type_of_order (category, 4 distinct): ['Snack ', 'Meal ', 'Drinks ', 'Buffet ']
Type_of_vehicle (category, 4 distinct): ['motorcycle ', 'scooter ', 'electric_scooter ', 'bicycle ']
'''

TARGET = CuratedTarget(raw_name="Time_taken(min)", task_type=SupervisedTask.REGRESSION)
