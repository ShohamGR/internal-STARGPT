from tabstar_paper.datasets.curation_objects import CuratedTarget
from tabstar_paper.datasets.objects import SupervisedTask

'''
Dataset Name: in_vehicle_coupon_recommendation
====
Examples: 12684
====
URL: https://www.openml.org/search?type=data&id=46937
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
- **Original Data Source:** https://doi.org/10.24432/C5GS4P
- **Reference (please cite)**: Wang, Tong, et al. 'A bayesian framework for learning rule sets for interpretable classification.' Journal of Machine Learning Research 18.70 (2017): 1-37. https://www.jmlr.org/papers/v18/16-003.html
- **Dataset Year:** 2017
- **Dataset Description:** see the reference and the original data source for details.

#### Curation comments by the TabArena team (for code see the [page of the study](https://tabarena.ai/data-tabular-ml-iid-study)):
- We renamed the target feature and its value to be more descriptive.
- We removed text data from the "time" feature to make it numeric.
- We drop constant columns.
- We fixed a typo in the feature names. 
- Anomaly: the data has many binned numeric features that are treated as categorical features with text describing the bins.
- Anomaly: the numeric features in the dataset are low-cardinality (<25)
====
Target Variable: AcceptCoupon (category, 2 distinct): ['Yes', 'No']
====
Features:

destination (category, 3 distinct): ['No Urgent Place', 'Home', 'Work']
passenger (category, 4 distinct): ['Alone', 'Friend(s)', 'Partner', 'Kid(s)']
weather (category, 3 distinct): ['Sunny', 'Snowy', 'Rainy']
temperature (uint8, 3 distinct): ['80', '55', '30']
time (uint8, 5 distinct): ['18', '7', '10', '14', '22']
coupon (object, 5 distinct): ['Coffee House', 'Restaurant(<20)', 'Carry out & Take away', 'Bar', 'Restaurant(20-50)']
expiration (category, 2 distinct): ['1d', '2h']
gender (category, 2 distinct): ['Female', 'Male']
age (category, 8 distinct): ['21', '26', '31', '50plus', '36', '41', '46', 'below21']
maritalStatus (category, 5 distinct): ['Married partner', 'Single', 'Unmarried partner', 'Divorced', 'Widowed']
has_children (category, 2 distinct): ['0', '1']
education (category, 6 distinct): ['Some college - no degree', 'Bachelors degree', 'Graduate degree (Masters or Doctorate)', 'Associates degree', 'High School Graduate', 'Some High School']
occupation (category, 25 distinct): ['Unemployed', 'Student', 'Computer & Mathematical', 'Sales & Related', 'Education&Training&Library', 'Management', 'Office & Administrative Support', 'Arts Design Entertainment Sports & Media', 'Business & Financial', 'Retired']
income (category, 9 distinct): ['$25000 - $37499', '$12500 - $24999', '$37500 - $49999', '$100000 or More', '$50000 - $62499', 'Less than $12500', '$87500 - $99999', '$75000 - $87499', '$62500 - $74999']
car (category, 6 distinct): ['nan', 'Mazda5', 'Scooter and motorcycle', 'do not drive', 'Car that is too old to install Onstar :D', 'crossover']
Bar (category, 6 distinct): ['never', 'less1', '1~3', '4~8', 'gt8', 'nan']
CoffeeHouse (category, 6 distinct): ['less1', '1~3', 'never', '4~8', 'gt8', 'nan']
CarryAway (category, 6 distinct): ['1~3', '4~8', 'less1', 'gt8', 'never', 'nan']
RestaurantLessThan20 (category, 6 distinct): ['1~3', '4~8', 'less1', 'gt8', 'never', 'nan']
Restaurant20To50 (category, 6 distinct): ['less1', '1~3', 'never', '4~8', 'gt8', 'nan']
toCoupon_GEQ15min (category, 2 distinct): ['1', '0']
toCoupon_GEQ25min (category, 2 distinct): ['0', '1']
direction_same (category, 2 distinct): ['0', '1']
direction_opp (category, 2 distinct): ['1', '0']
'''

TARGET = CuratedTarget(raw_name="AcceptCoupon", task_type=SupervisedTask.BINARY)
