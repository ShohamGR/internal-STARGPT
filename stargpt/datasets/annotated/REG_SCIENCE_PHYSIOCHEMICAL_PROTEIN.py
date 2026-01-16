from tabstar_paper.datasets.curation_objects import CuratedTarget
from tabstar_paper.datasets.objects import SupervisedTask

'''
Dataset Name: physiochemical_protein
====
Examples: 45730
====
URL: https://www.openml.org/search?type=data&id=46949
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
- **Original Data Source:** https://doi.org/10.24432/C5QW3H
- **Reference (please cite)**: Rana, Prashant. 'Physicochemical Properties of Protein Tertiary Structure.' UCI Machine Learning Repository, 2013, https://doi.org/10.24432/C5QW3H.
- **Dataset Year:** 2013
- **Dataset Description:** see the reference and the original data source for details.

#### Curation comments by the TabArena team (for code see the [page of the study](https://tabarena.ai/data-tabular-ml-iid-study)):
- We renamed the features to be more semantic meaningful.
====

# This is the description of dataset: 44963

====
Description: **Data Description**

This is a data set of Physicochemical Properties of Protein Tertiary Structure. The data set is taken from CASP 5-9. There are 45730 decoys and size varying from 0 to 21 armstrong.

The goal of the dataset is to predict the size of the residue for a tertiary protein structure (a 3d protein structure). Once linked in the protein chain, an individual amino acid is called a residue. The target feature is root mean square error of the residue.

**Attribute Description**

1. *RMSD* - size of the residue
2. *F1* - total surface area
3. *F2* - non polar exposed area
4. *F3* - fractional area of exposed non polar residue
5. *F4* - fractional area of exposed non polar part of residue
6. *F5* - molecular mass weighted exposed area
7. *F6* - average deviation from standard exposed area of residue
8. *F7* - Euclidian distance
9. *F8* - secondary structure penalty
10. *F9* - Spacial Distribution constraints (N,K Value)
====

Target Variable: ResidualSize (float64, 15903 distinct): ['0.0', '2.006', '1.787', '1.9', '2.055', '1.811', '1.937', '1.896', '2.012', '1.904']
====
Features:

TotalSurfaceArea (float64, 39916 distinct): ['13475.4', '5811.82', '4000.26', '14170.5', '20734.4', '4670.89', '15024.1', '4458.01', '11104.7', '5910.85']
NonPolarExposedArea (float64, 39863 distinct): ['4814.93', '1087.13', '1053.23', '1866.32', '2129.8', '3520.75', '7997.71', '1729.67', '2067.07', '3102.87']
FracExposedNonPolarResidue (float64, 20089 distinct): ['0.3573', '0.2718', '0.2485', '0.3118', '0.2663', '0.2795', '0.1812', '0.2873', '0.269', '0.2807']
FracExposedNonPolarPart (float64, 40374 distinct): ['168.55', '33.732', '52.5591', '186.407', '189.396', '46.7282', '174.306', '55.9897', '45.2729', '180.629']
MassWeightedExposedArea (float64, 41868 distinct): ['1877843.5474', '799977.1539', '569494.3892', '686984.9356', '2876946.082', '1937925.8075', '537119.5301', '604347.4122', '554911.2539', '2114837.6367']
AvgDeviationExposedArea (float64, 39155 distinct): ['227.605', '46.039', '65.2332', '66.6405', '214.666', '356.061', '211.148', '118.232', '157.29', '133.842']
EuclideanDistance (float64, 39450 distinct): ['4644.75', '4057.08', '3034.98', '1773.46', '1399.62', '4581.39', '2334.29', '4628.02', '3903.68', '2557.66']
SecondaryStructurePenalty (int64, 341 distinct): ['32', '17', '40', '36', '30', '41', '33', '27', '39', '38']
SpatialDistNK (float64, 37299 distinct): ['46.5464', '29.7563', '44.4892', '34.8833', '39.7659', '38.8321', '38.1176', '44.4197', '28.9962', '44.2712']
'''

CONTEXT = "Physiochemical Protein for Tertiary Structure"
TARGET = CuratedTarget(raw_name="ResidualSize", task_type=SupervisedTask.REGRESSION)