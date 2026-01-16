from tabstar_paper.datasets.curation_objects import CuratedTarget
from tabstar_paper.datasets.objects import SupervisedTask

'''
Dataset Name: miami_housing
====
Examples: 13776
====
URL: https://www.openml.org/search?type=data&id=46942
====
Description: This dataset was curated for [TabArena](https://tabarena.ai/) by the TabArena team
as part of the [TabArena Tabular ML IID Study](https://tabarena.ai/data-tabular-ml-iid-study).
For more details on the study, see our [paper](https://tabarena.ai/paper-tabular-ml-iid-study).

**Dataset Focus**: This dataset shall be used for evaluating predictive machine
learning models for independent and identically distributed tabular data. The
intended task is regression.

---
#### Dataset Metadata
- **Licence:** CC BY-NC-SA
- **Original Data Source:** https://www.openml.org/search?type=data&id=43093&sort=runs&status=active
- **Reference (please cite)**: Bourassa, Steven C., et al. 'Big data, accessibility and urban house prices.' Urban Studies 58.15 (2021): 3176-3195. https://doi.org/10.1177/0042098020982508
- **Dataset Year:** 2016
- **Dataset Description:** see the reference and the original data source for details.

#### Curation comments by the TabArena team (for code see the [page of the study](https://tabarena.ai/data-tabular-ml-iid-study)):
- We drop duplicated homes (non-unique identifiers and location) (~1% of the data).
- We drop the ID column "PARCELNO". 
- Anomaly: while the original paper describes data with 57k samples, we only have 13k on OpenML or Kaggle.
====

# This is the description of dataset: 44983

====
Description: **Data description**

The dataset contains information on 13,932 single-family homes sold in Miami .
Content.

The goal is to predict the sale price.

**Attribute description**

The dataset contains the following columns:

  * PARCELNO: unique identifier for each property. About 1% appear multiple times.
  * SALE_PRC: sale price ($)
  * LND_SQFOOT: land area (square feet)
  * TOTLVGAREA: floor area (square feet)
  * SPECFEATVAL: value of special features (e.g., swimming pools) ($)
  * RAIL_DIST: distance to the nearest rail line (an indicator of noise) (feet)
  * OCEAN_DIST: distance to the ocean (feet)
  * WATER_DIST: distance to the nearest body of water (feet)
  * CNTR_DIST: distance to the Miami central business district (feet)
  * SUBCNTR_DI: distance to the nearest subcenter (feet)
  * HWY_DIST: distance to the nearest highway (an indicator of noise) (feet)
  * age: age of the structure
  * avno60plus: dummy variable for airplane noise exceeding an acceptable level
  * structure_quality: quality of the structure
  * month_sold: sale month in 2016 (1 = jan)
  * LATITUDE
  * LONGITUDE
====
Target Variable: SALE_PRC (float64, 2106 distinct): ['250000.0', '300000.0', '260000.0', '270000.0', '290000.0', '280000.0', '265000.0', '350000.0', '285000.0', '210000.0']
====
Features:

LATITUDE (float64, 13776 distinct): ['25.905', '25.7183', '25.7028', '25.9375', '25.8752', '25.6915', '25.502', '25.5848', '25.7531', '25.7401']
LONGITUDE (float64, 13776 distinct): ['-80.1688', '-80.41', '-80.455', '-80.3037', '-80.2067', '-80.3521', '-80.4134', '-80.3215', '-80.3068', '-80.4352']
LND_SQFOOT (float64, 4696 distinct): ['7500.0', '5000.0', '6000.0', '7875.0', '8250.0', '8000.0', '15000.0', '5500.0', '5250.0', '10000.0']
TOT_LVG_AREA (float64, 2978 distinct): ['3079.0', '3199.0', '2176.0', '1440.0', '1701.0', '2578.0', '2193.0', '2091.0', '1871.0', '2514.0']
SPEC_FEAT_VAL (float64, 7583 distinct): ['0.0', '550.0', '440.0', '4800.0', '1200.0', '3200.0', '2240.0', '2200.0', '2460.0', '495.0']
RAIL_DIST (float64, 13235 distinct): ['50.0', '49.9', '7970.2', '14690.8', '7529.5', '16135.4', '2539.6', '77.9', '1675.8', '13911.1']
OCEAN_DIST (float64, 13617 distinct): ['28968.2', '27798.4', '15069.8', '24724.8', '55025.2', '28901.2', '24744.6', '58526.9', '24235.6', '48122.1']
WATER_DIST (float64, 13218 distinct): ['0.0', '7.2', '10267.8', '523.4', '6252.7', '10.5', '11.0', '2889.2', '2750.0', '3644.1']
CNTR_DIST (float64, 13682 distinct): ['43025.9', '46027.1', '68958.6', '35177.4', '68317.7', '38072.8', '56955.1', '82126.2', '28715.5', '38799.3']
SUBCNTR_DI (float64, 13642 distinct): ['60514.7', '45214.2', '7143.3', '45441.5', '32079.8', '52022.9', '30277.7', '7281.3', '53622.0', '49014.8']
HWY_DIST (float64, 13213 distinct): ['2140.8', '8165.7', '3751.1', '1445.8', '4589.9', '4007.4', '4550.7', '3212.1', '9580.4', '997.3']
age (uint8, 96 distinct): ['0', '16', '26', '21', '11', '36', '12', '23', '10', '31']
avno60plus (category, 2 distinct): ['0.0', '1.0']
month_sold (uint8, 12 distinct): ['6', '8', '5', '4', '3', '9', '7', '12', '11', '10']
structure_quality (uint8, 5 distinct): ['4', '2', '5', '1', '3']
'''

CONTEXT = "Family Houses sold in Miami"
TARGET = CuratedTarget(raw_name="SALE_PRC", task_type=SupervisedTask.REGRESSION)