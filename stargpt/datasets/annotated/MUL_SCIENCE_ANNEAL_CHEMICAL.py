from tabstar_paper.datasets.curation_objects import CuratedTarget
from tabstar_paper.datasets.objects import SupervisedTask

'''
Dataset Name: anneal
====
Examples: 898
====
URL: https://www.openml.org/search?type=data&id=46906
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
- **Original Data Source:** https://doi.org/10.24432/C5RW2F
- **Reference (please cite)**: 'Annealing.' UCI Machine Learning Repository,  https://doi.org/10.24432/C5RW2F.
- **Dataset Year:** 1990
- **Dataset Description:** see the reference and the original data source for details.

#### Curation comments by the TabArena team (for code see the [page of the study](https://tabarena.ai/data-tabular-ml-iid-study)):
- We combined train and test data into a single dataset.
- We replaced '?' with 'not_applicable' as described in the metadata.
- We renamed some features to remove '/' characters.
- Anomaly: In the original data, class 4 is in the metadata, but there are no samples in this class.
====

# This is the description of dataset: 2

Description: **Author**: Unknown. Donated by David Sterling and Wray Buntine  

**Source**: [UCI](https://archive.ics.uci.edu/ml/datasets/Annealing) - 1990  

**Please cite**: [UCI](https://archive.ics.uci.edu/ml/citation_policy.html)  



The original Annealing dataset from UCI. The exact meaning of the features and classes is largely unknown. Annealing, in metallurgy and materials science, is a heat treatment that alters the physical and sometimes chemical properties of a material to increase its ductility and reduce its hardness, making it more workable. It involves heating a material to above its recrystallization temperature, maintaining a suitable temperature, and then cooling. (Wikipedia)



### Attribute Information:

     1. family:          --,GB,GK,GS,TN,ZA,ZF,ZH,ZM,ZS

     2. product-type:    C, H, G

     3. steel:           -,R,A,U,K,M,S,W,V

     4. carbon:          continuous

     5. hardness:        continuous

     6. temper_rolling:  -,T

     7. condition:       -,S,A,X

     8. formability:     -,1,2,3,4,5

     9. strength:        continuous

    10. non-ageing:      -,N

    11. surface-finish:  P,M,-

    12. surface-quality: -,D,E,F,G

    13. enamelability:   -,1,2,3,4,5

    14. bc:              Y,-

    15. bf:              Y,-

    16. bt:              Y,-

    17. bw/me:           B,M,-

    18. bl:              Y,-

    19. m:               Y,-

    20. chrom:           C,-

    21. phos:            P,-

    22. cbond:           Y,-

    23. marvi:           Y,-

    24. exptl:           Y,-

    25. ferro:           Y,-

    26. corr:            Y,-

    27. blue/bright/varn/clean:          B,R,V,C,-

    28. lustre:          Y,-

    29. jurofm:          Y,-

    30. s:               Y,-

    31. p:               Y,-

    32. shape:           COIL, SHEET

    33. thick:           continuous

    34. width:           continuous

    35. len:             continuous

    36. oil:             -,Y,N

    37. bore:            0000,0500,0600,0760

    38. packing: -,1,2,3

    classes:        1,2,3,4,5,U

  

    -- The '-' values are actually 'not_applicable' values rather than

       'missing_values' (and so can be treated as legal discrete

       values rather than as showing the absence of a discrete value).

====       
Target Variable: classes (category, 5 distinct): ['3', '2', '5', 'U', '1']
====
Features:

family (category, 3 distinct): ['not_applicable', 'TN', 'ZS']
product-type (category, 1 distinct): ['C']
steel (category, 8 distinct): ['A', 'R', 'not_applicable', 'K', 'M', 'W', 'V', 'S']
carbon (uint8, 10 distinct): ['0', '55', '45', '65', '6', '70', '4', '8', '10', '3']
hardness (uint8, 7 distinct): ['0', '45', '85', '50', '60', '70', '80']
temper_rolling (category, 2 distinct): ['not_applicable', 'T']
condition (category, 3 distinct): ['S', 'not_applicable', 'A']
formability (category, 5 distinct): ['2', 'not_applicable', '3', '1', '5']
strength (int64, 8 distinct): ['0', '310', '500', '600', '350', '400', '300', '700']
non-ageing (category, 2 distinct): ['not_applicable', 'N']
surface-finish (category, 2 distinct): ['not_applicable', 'P']
surface-quality (category, 5 distinct): ['E', 'not_applicable', 'G', 'F', 'D']
enamelability (category, 3 distinct): ['not_applicable', '2', '1']
bc (category, 2 distinct): ['not_applicable', 'Y']
bf (category, 2 distinct): ['not_applicable', 'Y']
bt (category, 2 distinct): ['not_applicable', 'Y']
bw_me (category, 3 distinct): ['not_applicable', 'B', 'M']
bl (category, 2 distinct): ['not_applicable', 'Y']
m (category, 1 distinct): ['not_applicable']
chrom (category, 2 distinct): ['not_applicable', 'C']
phos (category, 2 distinct): ['not_applicable', 'P']
cbond (category, 2 distinct): ['not_applicable', 'Y']
marvi (category, 1 distinct): ['not_applicable']
exptl (category, 2 distinct): ['not_applicable', 'Y']
ferro (category, 2 distinct): ['not_applicable', 'Y']
corr (category, 1 distinct): ['not_applicable']
blue_bright_varn_clean (category, 4 distinct): ['not_applicable', 'B', 'C', 'V']
lustre (category, 2 distinct): ['not_applicable', 'Y']
jurofm (category, 1 distinct): ['not_applicable']
s (category, 1 distinct): ['not_applicable']
p (category, 1 distinct): ['not_applicable']
shape (category, 2 distinct): ['SHEET', 'COIL']
thick (float64, 50 distinct): ['0.7', '1.6', '0.699', '0.6', '0.8', '3.2', '0.3', '1.2', '0.4', '1.599']
width (float64, 68 distinct): ['610.0', '1320.0', '609.9', '1220.0', '1300.0', '900.0', '20.0', '50.0', '150.0', '1250.0']
len (int64, 24 distinct): ['0', '762', '4880', '612', '4170', '761', '3000', '301', '1', '150']
oil (category, 3 distinct): ['not_applicable', 'Y', 'N']
bore (category, 3 distinct): ['0', '600', '500']
packing (category, 3 distinct): ['not_applicable', '3', '2']
'''

CONTEXT = "Anneal Chemical Process"
TARGET = CuratedTarget(raw_name="classes", task_type=SupervisedTask.MULTICLASS)