from tabstar_paper.datasets.curation_objects import CuratedTarget
from tabstar_paper.datasets.objects import SupervisedTask

'''
Dataset Name: hazelnut-spread-contaminant-detection
====
Examples: 2400
====
URL: https://www.openml.org/search?type=data&id=46930
====
Description: This dataset was curated for [TabArena](https://tabarena.ai/) by the TabArena team
as part of the [TabArena Tabular ML IID Study](https://tabarena.ai/data-tabular-ml-iid-study).
For more details on the study, see our [paper](https://tabarena.ai/paper-tabular-ml-iid-study).

**Dataset Focus**: This dataset shall be used for evaluating predictive machine
learning models for independent and identically distributed tabular data. The
intended task is classification.

---
#### Dataset Metadata
- **Licence:** CC BY-SA
- **Original Data Source:** https://www.openml.org/search?type=data&status=active&id=45538
- **Reference (please cite)**: Ricci, Marco, et al. 'Machine-learning-based microwave sensing: A case study for the food industry.' IEEE Journal on Emerging and Selected Topics in Circuits and Systems 11.3 (2021): 503-514. https://doi.org/10.1109/JETCAS.2021.3097699; Urbinati, Luca, et al. 'A machine-learning based microwave sensing approach to food contaminant detection.' 2020 IEEE International Symposium on Circuits and Systems (ISCAS). IEEE, 2020. https://doi.org/10.1109/ISCAS45731.2020.9181293
- **Dataset Year:** 2020
- **Dataset Description:** see the reference and the original data source for details.

#### Curation comments by the TabArena team (for code see the [page of the study](https://tabarena.ai/data-tabular-ml-iid-study)):
- We select a features from a single frequency (10 GHz) as the authors also only considered this frequency for the final experiments.
- Anomaly: we use the publicly available dataset state, which is without preprocessing. Moreover, we were not able to inverse the original ordinal encoding of the label.
====
Target Variable: Contaminated (category, 2 distinct): ['0', '1']
====
Features:

s12 (float64, 1492 distinct): ['-0.0017', '-0.0025', '-0.0024', '-0.0021', '-0.0022', '-0.0023', '-0.002', '-0.0025', '-0.0025', '-0.0019']
s13 (float64, 1409 distinct): ['-0.0007', '-0.0007', '-0.0006', '-0.0007', '-0.0008', '-0.0007', '-0.0007', '-0.0007', '-0.0006', '-0.0006']
s14 (float64, 1727 distinct): ['-0.0029', '-0.0027', '-0.0013', '-0.0012', '-0.0009', '-0.0028', '-0.0012', '-0.0045', '-0.0028', '-0.003']
s15 (float64, 1734 distinct): ['0.0009', '-0.0019', '0.0009', '-0.002', '0.0008', '-0.002', '-0.0016', '-0.0018', '0.0006', '-0.0018']
s16 (float64, 1627 distinct): ['-0.0003', '0.0003', '-0.0003', '-0.0', '-0.0006', '0.0004', '0.0001', '-0.0002', '-0.0', '-0.0002']
s21 (float64, 1757 distinct): ['0.0021', '0.0014', '0.0023', '-0.0002', '0.0013', '0.0007', '0.0011', '0.0008', '0.0023', '0.0009']
s23 (float64, 1661 distinct): ['-0.0011', '-0.001', '-0.0011', '-0.0008', '-0.001', '-0.001', '-0.0012', '0.0011', '-0.0012', '-0.0009']
s24 (float64, 1466 distinct): ['-0.0024', '-0.0024', '-0.0025', '-0.0023', '-0.0024', '-0.0025', '-0.0004', '-0.0026', '0.0001', '-0.0025']
s25 (float64, 1455 distinct): ['-0.0001', '-0.0002', '-0.0003', '-0.0005', '-0.0002', '-0.0001', '-0.001', '-0.0', '-0.0003', '-0.0007']
s26 (float64, 1057 distinct): ['-0.0002', '-0.0002', '-0.0004', '-0.0002', '-0.0002', '0.0001', '-0.0002', '-0.0001', '-0.0002', '-0.0']
s31 (float64, 1554 distinct): ['-0.0025', '-0.0024', '-0.0022', '-0.0024', '-0.0023', '-0.0024', '-0.0024', '-0.0026', '-0.0026', '-0.0025']
s32 (float64, 1596 distinct): ['-0.004', '-0.0042', '-0.004', '-0.005', '-0.005', '-0.004', '-0.0051', '-0.0042', '-0.0046', '-0.0041']
s34 (float64, 1504 distinct): ['-0.0015', '-0.0', '-0.0011', '-0.0006', '-0.0008', '-0.0001', '-0.0004', '-0.0015', '0.0003', '-0.0006']
s35 (float64, 813 distinct): ['-0.0005', '-0.0005', '-0.0005', '-0.0006', '-0.0005', '-0.0006', '-0.0005', '-0.0006', '-0.0005', '-0.0006']
s36 (float64, 1248 distinct): ['0.0024', '0.0025', '0.0022', '0.0024', '0.0023', '0.0022', '0.0026', '0.0025', '0.0026', '0.0025']
s41 (float64, 1037 distinct): ['0.0003', '0.0002', '0.0003', '0.0003', '0.0003', '0.0003', '0.0003', '0.0003', '-0.0003', '0.0003']
s42 (float64, 1058 distinct): ['0.0003', '0.0003', '0.0004', '-0.0003', '0.0003', '-0.0001', '0.0003', '-0.0003', '0.0003', '-0.0002']
s43 (float64, 966 distinct): ['0.0003', '0.0003', '0.0005', '0.0003', '0.0003', '0.0003', '0.0005', '0.0004', '0.0002', '0.0002']
s45 (float64, 1313 distinct): ['0.0023', '0.0022', '0.0016', '0.0022', '0.0021', '0.0016', '0.0017', '0.0021', '0.0017', '0.0021']
s46 (float64, 933 distinct): ['0.0022', '0.0021', '0.0021', '0.0021', '0.0022', '0.0021', '0.0021', '0.002', '0.0021', '0.0022']
s51 (float64, 882 distinct): ['-0.0012', '-0.0012', '-0.0013', '-0.0012', '-0.0012', '-0.0013', '-0.0013', '-0.0012', '-0.0013', '-0.0013']
s52 (float64, 899 distinct): ['-0.0008', '-0.0008', '-0.0008', '-0.001', '-0.0007', '-0.0007', '-0.0006', '-0.0007', '-0.0007', '-0.0009']
s53 (float64, 1190 distinct): ['0.0023', '0.0023', '0.0021', '0.0021', '0.0023', '0.0023', '0.0022', '0.0022', '0.0021', '0.0023']
s54 (float64, 1319 distinct): ['-0.002', '-0.0019', '-0.0019', '-0.002', '-0.0021', '-0.0022', '-0.0018', '-0.002', '-0.0021', '-0.0021']
s56 (float64, 861 distinct): ['-0.0022', '-0.0022', '-0.0022', '-0.0022', '-0.0022', '-0.0022', '-0.0022', '-0.0022', '-0.0021', '-0.0022']
s61 (float64, 781 distinct): ['-0.0011', '-0.0008', '-0.0011', '-0.0011', '-0.0011', '-0.0011', '-0.0011', '-0.001', '-0.0011', '-0.0011']
s62 (float64, 1201 distinct): ['0.0004', '0.0006', '0.0006', '0.0006', '0.0006', '0.0005', '0.0006', '0.0006', '0.0006', '0.0006']
s63 (float64, 1072 distinct): ['0.0013', '0.0012', '0.0013', '0.0012', '0.0012', '0.0012', '0.0012', '0.0012', '0.0013', '0.0013']
s64 (float64, 1156 distinct): ['0.0021', '0.0021', '0.0018', '0.0018', '0.0017', '0.0018', '0.0018', '0.0017', '0.002', '0.0021']
s65 (float64, 1359 distinct): ['-0.0017', '-0.0017', '-0.0022', '-0.0018', '-0.0017', '-0.0017', '-0.0016', '-0.0016', '-0.0019', '-0.0017']

'''

CONTEXT = "Microwave sensing data for contaminant detection in hazelnut spread"
TARGET = CuratedTarget(raw_name="Contaminated", task_type=SupervisedTask.BINARY)
