from tabstar_paper.datasets.curation_objects import CuratedTarget
from tabstar_paper.datasets.objects import SupervisedTask

'''
Dataset Name: seismic-bumps
====
Examples: 2584
====
URL: https://www.openml.org/search?type=data&id=46956
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
- **Original Data Source:** https://doi.org/10.24432/C5W902
- **Reference (please cite)**: Sikora, Marek, and ukasz Wrobel. 'Application of rule induction algorithms for analysis of data collected by seismic hazard monitoring systems in coal mines.' Archives of Mining Sciences 55.1 (2010): 91-114. http://yadda.icm.edu.pl/baztech/element/bwmeta1.element.baztech-article-BPZ5-0008-0008
- **Dataset Year:** 2013
- **Dataset Description:** see the reference and the original data source for details.

#### Curation comments by the TabArena team (for code see the [page of the study](https://tabarena.ai/data-tabular-ml-iid-study)):
- We renamed the target feature and reversed the values' ordinal encoding.
- We drop the constant columns "nbumps6", "nbumps7", and "nbumps89".
====
Target Variable: HighEnergySeismicBump (category, 2 distinct): ['No', 'Yes']
====
Features:

seismic (category, 2 distinct): ['a', 'b']
seismoacoustic (category, 3 distinct): ['a', 'b', 'c']
shift (category, 2 distinct): ['W', 'N']
genergy (float64, 2212 distinct): ['7400.0', '6790.0', '3610.0', '11230.0', '19420.0', '15300.0', '20160.0', '5640.0', '12950.0', '23040.0']
gpuls (float64, 1128 distinct): ['19.0', '46.0', '213.0', '17.0', '133.0', '25.0', '53.0', '262.0', '24.0', '402.0']
gdenergy (float64, 334 distinct): ['-14.0', '-32.0', '-7.0', '-31.0', '-10.0', '-38.0', '-42.0', '-20.0', '-40.0', '-21.0']
gdpuls (float64, 292 distinct): ['0.0', '-32.0', '-14.0', '6.0', '-28.0', '2.0', '-42.0', '-2.0', '-6.0', '-40.0']
ghazard (object, 3 distinct): ['a', 'b', 'c']
nbumps (uint8, 10 distinct): ['0', '1', '2', '3', '4', '5', '6', '8', '7', '9']
nbumps2 (uint8, 7 distinct): ['0', '1', '2', '3', '4', '5', '8']
nbumps3 (uint8, 7 distinct): ['0', '1', '2', '3', '4', '5', '7']
nbumps4 (uint8, 4 distinct): ['0', '1', '2', '3']
nbumps5 (uint8, 2 distinct): ['0', '1']
energy (float64, 242 distinct): ['0.0', '4000.0', '3000.0', '2000.0', '300.0', '400.0', '1000.0', '5000.0', '200.0', '600.0']
maxenergy (float64, 33 distinct): ['0.0', '3000.0', '2000.0', '4000.0', '1000.0', '5000.0', '300.0', '400.0', '6000.0', '20000.0']
'''

CONTEXT = "Seismic bumps hazard in coal mines"
TARGET = CuratedTarget(raw_name="HighEnergySeismicBump", task_type=SupervisedTask.BINARY)
