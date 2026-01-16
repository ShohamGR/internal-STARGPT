from tabstar_paper.datasets.curation_objects import CuratedTarget
from tabstar_paper.datasets.objects import SupervisedTask

'''
Dataset Name: houses
====
Examples: 20640
====
URL: https://www.openml.org/search?type=data&id=46934
====
Description: This dataset was curated for [TabArena](https://tabarena.ai/) by the TabArena team
as part of the [TabArena Tabular ML IID Study](https://tabarena.ai/data-tabular-ml-iid-study).
For more details on the study, see our [paper](https://tabarena.ai/paper-tabular-ml-iid-study).

**Dataset Focus**: This dataset shall be used for evaluating predictive machine
learning models for independent and identically distributed tabular data. The
intended task is regression.

---
#### Dataset Metadata
- **Licence:** Public
- **Original Data Source:** https://lib.stat.cmu.edu/datasets/
- **Reference (please cite)**: Pace, R. Kelley, and Ronald Barry. 'Sparse spatial autoregressions.' Statistics & Probability Letters 33.3 (1997): 291-297. https://doi.org/10.1016/S0167-7152(96)00140-X
- **Dataset Year:** 1990
- **Dataset Description:** see the reference and the original data source for details.

#### Curation comments by the TabArena team (for code see the [page of the study](https://tabarena.ai/data-tabular-ml-iid-study)):
- We kept the latitude and longitude features as they are.
- We log scaled the target variable (with base e) as intended by the original task.
- Anomaly: As always, we randomly shuffle the data before uploading. If one does not randomly shuffle the data, there would be a distribution shift from the longitude and latitude based on the original order of data samples.
====
Description of 44977

Information on the variables was collected using all the block groups in California from the 1990 Census. In this sample a block group on average includes 1425.5 individuals living in a geographically compact area. Naturally, the geographical area included varies inversely with the population density. Distances among the centroids of each block group were computed as measured in latitude and longitude. All the block groups reporting zero entries for the independent and dependent variables were excluded. The final data contained 20,640 observations on 9 variables. 

Each row in the dataset represents one census block group. A block group is the smallest geographical unit for which the U.S. Census Bureau publishes sample data (a block group typically has a population of 600 to 3,000 people).

The goal of the dataset is to predict the median house value. The original dataset description advised to predict the value using logarithmic transform.


**Attribute Description**

Census block group describing features:

1. *longitude* 
2. *latitude*
3. *housingMedianAge*
4. *totalRooms*
5. *totalBedrooms*
6. *population*
7. *households*
8. *medianIncome*
9. *medianHouseValue* - target feature
====
Target Variable: LnMedianHouseValue (float64, 3842 distinct): ['13.1224', '11.8314', '11.9984', '11.6307', '12.1415', '12.3239', '12.7657', '11.3794', '12.5245', '11.9184']
====
Features:

MedianIncome (float64, 12928 distinct): ['15.0001', '3.125', '2.875', '2.625', '4.125', '3.875', '3.0', '3.375', '3.625', '4.0']
HousingMedianAge (uint8, 52 distinct): ['52', '36', '35', '16', '17', '34', '26', '33', '18', '25']
TotalRooms (float64, 5926 distinct): ['1527.0', '1613.0', '1582.0', '2127.0', '1717.0', '1722.0', '1703.0', '1471.0', '1607.0', '2053.0']
TotalBedrooms (float64, 1928 distinct): ['280.0', '331.0', '345.0', '343.0', '394.0', '393.0', '328.0', '309.0', '348.0', '314.0']
Population (float64, 3888 distinct): ['891.0', '1227.0', '850.0', '761.0', '1052.0', '825.0', '1005.0', '782.0', '999.0', '1098.0']
Households (float64, 1815 distinct): ['306.0', '335.0', '386.0', '282.0', '429.0', '375.0', '284.0', '297.0', '340.0', '362.0']
Latitude (float64, 862 distinct): ['34.06', '34.05', '34.08', '34.07', '34.04', '34.09', '34.02', '34.1', '34.03', '33.93']
Longitude (float64, 844 distinct): ['-118.31', '-118.3', '-118.29', '-118.27', '-118.32', '-118.28', '-118.35', '-118.36', '-118.19', '-118.25']

'''

CONTEXT = "California Housing Prices"
TARGET = CuratedTarget(raw_name="LnMedianHouseValue", task_type=SupervisedTask.REGRESSION)
