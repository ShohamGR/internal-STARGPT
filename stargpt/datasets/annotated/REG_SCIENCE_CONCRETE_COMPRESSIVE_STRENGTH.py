from tabstar_paper.datasets.curation_objects import CuratedTarget
from tabstar_paper.datasets.objects import SupervisedTask

'''
Dataset Name: concrete_compressive_strength
====
Examples: 1030
====
URL: https://www.openml.org/search?type=data&id=46917
====
Description: This dataset was curated for [TabArena](https://tabarena.ai/) by the TabArena team
as part of the [TabArena Tabular ML IID Study](https://tabarena.ai/data-tabular-ml-iid-study).
For more details on the study, see our [paper](https://tabarena.ai/paper-tabular-ml-iid-study).

**Dataset Focus**: This dataset shall be used for evaluating predictive machine
learning models for independent and identically distributed tabular data. The
intended task is regression.

---
#### Dataset Metadata
- **Licence:** CC BY 4.0
- **Original Data Source:** https://doi.org/10.24432/C5PK67
- **Reference (please cite)**: Yeh, I-C. 'Modeling of strength of high-performance concrete using artificial neural networks.' Cement and Concrete research 28.12 (1998): 1797-1808. https://doi.org/10.1016/S0008-8846(98)00165-3
- **Dataset Year:** 1998
- **Dataset Description:** see the reference and the original data source for details.

#### Curation comments by the TabArena team (for code see the [page of the study](https://tabarena.ai/data-tabular-ml-iid-study)):
- We rename features to be shorter while similar to the original names.
====

# This is the description of dataset: 44959

Description: **Data Description**

Concrete is the most important material in civil engineering. The concrete compressive strength is a highly nonlinear function of age and ingredients. Each instance represents a description, different features of concrete instance, including its compressive strength. The latter can be predicted using the other features about the concrete.

**Attribute Description**

1. *cement* - amount in kg in a m3 mixture
2. *blast_furnace_slag* - amount in kg in a m3 mixture
3. *fly_ash* - amount in kg in a m3 mixture
4. *water* - amount in kg in a m3 mixture
5. *superplasticizer* - amount in kg in a m3 mixture
6. *coarse_aggregate* - amount in kg in a m3 mixture
7. *fine_aggregate* - amount in kg in a m3 mixture
8. *age* - age in days (1 - 365)
9. *strength* - in MPa, target feature

====
Target Variable: ConcreteCompressiveStrength (float64, 938 distinct): ['33.3982', '77.2972', '31.3505', '71.2987', '35.3012', '79.2966', '25.1797', '65.1969', '18.1263', '64.3005']
====
Features:

Cement (float64, 280 distinct): ['362.6', '425.0', '251.37', '310.0', '446.0', '475.0', '331.0', '250.0', '349.0', '387.0']
BlastFurnaceSlag (float64, 187 distinct): ['0.0', '189.0', '106.3', '24.0', '20.0', '145.0', '19.0', '22.0', '26.0', '190.0']
FlyAsh (float64, 163 distinct): ['0.0', '141.0', '118.27', '79.0', '94.0', '98.75', '125.18', '100.52', '121.62', '95.69']
Water (float64, 205 distinct): ['192.0', '228.0', '185.7', '203.5', '186.0', '162.0', '164.9', '153.5', '185.0', '178.0']
Superplasticizer (float64, 155 distinct): ['0.0', '8.0', '11.6', '7.0', '6.0', '10.0', '9.0', '16.5', '11.0', '9.88']
CoarseAggregate (float64, 284 distinct): ['932.0', '852.1', '944.7', '968.0', '1125.0', '1047.0', '967.0', '942.0', '974.0', '822.0']
FineAggregate (float64, 304 distinct): ['594.0', '755.8', '670.0', '613.0', '801.0', '887.1', '746.6', '712.0', '845.0', '750.0']
Age (int64, 14 distinct): ['28', '3', '7', '56', '14', '90', '100', '180', '91', '365']
'''

CONTEXT = "Concrete Compressive Strength"
TARGET = CuratedTarget(raw_name="ConcreteCompressiveStrength", task_type=SupervisedTask.REGRESSION)