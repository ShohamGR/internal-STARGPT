from tabstar_paper.datasets.curation_objects import CuratedTarget
from tabstar_paper.datasets.objects import SupervisedTask

'''
Dataset Name: customer_satisfaction_in_airline
====
Examples: 129880
====
URL: https://www.openml.org/search?type=data&id=46920
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
- **Original Data Source:** https://www.kaggle.com/datasets/yakhyojon/customer-satisfaction-in-airline
- **Reference (please cite)**: Kaggle User Yakhyojon. 'Customer Satisfaction in Airline.' Kaggle, 2023, https://www.kaggle.com/datasets/yakhyojon/customer-satisfaction-in-airline.
- **Dataset Year:** 2023
- **Dataset Description:** see the reference and the original data source for details.

The data is for a sample size of 129,880 customers. It includes data points such as class, flight distance, and inflight entertainment to be used to predict whether a customer will be satisfied with their flight experience.


#### Curation comments by the TabArena team (for code see the [page of the study](https://tabarena.ai/data-tabular-ml-iid-study)):
- We removed whitespaces and special characters from the feature names.
====
Target Variable: satisfaction (category, 2 distinct): ['satisfied', 'dissatisfied']
====
Features:

CustomerType (category, 2 distinct): ['Loyal Customer', 'disloyal Customer']
Age (uint8, 75 distinct): ['39', '25', '40', '44', '41', '42', '43', '45', '23', '22']
TypeofTravel (category, 2 distinct): ['Business travel', 'Personal Travel']
Class (category, 3 distinct): ['Business', 'Eco', 'Eco Plus']
FlightDistance (int64, 5398 distinct): ['1963', '1812', '1639', '1789', '1981', '1759', '1766', '1748', '1769', '2022']
Seatcomfort (category, 6 distinct): ['3', '2', '4', '1', '5', '0']
DepartureArrivaltimeconvenient (category, 6 distinct): ['4', '5', '3', '2', '1', '0']
Foodanddrink (category, 6 distinct): ['3', '4', '2', '1', '5', '0']
Gatelocation (category, 6 distinct): ['3', '4', '2', '1', '5', '0']
Inflightwifiservice (category, 6 distinct): ['4', '5', '3', '2', '1', '0']
Inflightentertainment (category, 6 distinct): ['4', '5', '3', '2', '1', '0']
Onlinesupport (category, 6 distinct): ['4', '5', '3', '2', '1', '0']
EaseofOnlinebooking (category, 6 distinct): ['4', '5', '3', '2', '1', '0']
Onboardservice (category, 6 distinct): ['4', '5', '3', '2', '1', '0']
Legroomservice (category, 6 distinct): ['4', '5', '3', '2', '1', '0']
Baggagehandling (category, 5 distinct): ['4', '5', '3', '2', '1']
Checkinservice (category, 6 distinct): ['4', '3', '5', '2', '1', '0']
Cleanliness (category, 6 distinct): ['4', '5', '3', '2', '1', '0']
Onlineboarding (uint8, 6 distinct): ['4', '3', '5', '2', '1', '0']
DepartureDelayinMinutes (int64, 466 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
ArrivalDelayinMinutes (float64, 472 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
'''

TARGET = CuratedTarget(raw_name="satisfaction", task_type=SupervisedTask.BINARY)
