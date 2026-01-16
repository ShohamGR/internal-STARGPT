from tabstar_paper.datasets.curation_objects import CuratedTarget
from tabstar_paper.datasets.objects import SupervisedTask

'''
Dataset Name: splice
====
Examples: 3190
====
URL: https://www.openml.org/search?type=data&id=46958
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
- **Original Data Source:** https://doi.org/10.24432/C5M888
- **Reference (please cite)**: Towell, Geoffrey G., and Jude W. Shavlik. 'Knowledge-based artificial neural networks.' Artificial intelligence 70.1-2 (1994): 119-165. https://doi.org/10.1016/0004-3702(94)90105-8
- **Dataset Year:** 1991
- **Dataset Description:** see the reference and the original data source for details.

#### Curation comments by the TabArena team (for code see the [page of the study](https://tabarena.ai/data-tabular-ml-iid-study)):
- We tabularized the fixed-size DNA sequences from the original dataset based on the nucleotide position.
- We removed the instance ID.
- Anomaly: As always, we randomly shuffle the data before uploading. If one would not randomly shuffle the data, there would be a shift based on the original order of data samples.
====

# This is the description of dataset: 46
====
Description: **Author**: Genbank. Donated by G. Towell, M. Noordewier, and J. Shavlik  
**Source**: [UCI](https://archive.ics.uci.edu/ml/datasets/Molecular+Biology+(Splice-junction+Gene+Sequences))   
**Please cite**:  None  

Primate splice-junction gene sequences (DNA) with associated imperfect domain theory.
Splice junctions are points on a DNA sequence at which 'superfluous' DNA is removed during the process of protein creation in higher organisms. The problem posed in this dataset is to recognize, given a sequence of DNA, the boundaries between exons (the parts of the DNA sequence retained after splicing) and introns (the parts of the DNA sequence that are spliced out). This problem consists of two subtasks: recognizing exon/intron boundaries (referred to as EI sites), and recognizing intron/exon boundaries (IE sites). (In the biological community, IE borders are referred to a ''acceptors'' while EI borders are referred to as ''donors''.)

All examples taken from Genbank 64.1. Categories "ei" and "ie" include every "split-gene" for primates in Genbank 64.1. Non-splice examples taken from sequences known not to include a splicing site.
         
### Attribute Information 
>
              1   One of {n ei ie}, indicating the class.
              2   The instance name.
           3-62   The remaining 60 fields are the sequence, starting at 
                  position -30 and ending at position +30. Each of
                  these fields is almost always filled by one of 
                  {a, g, t, c}. Other characters indicate ambiguity among
                  the standard characters according to the following table:
    character: meaning
        D: A or G or T
        N: A or G or C or T
        S: C or G
        R: A or G

Notes:  
* Instance_name is an identifier and should be ignored for modelling
====
Target Variable: SiteType (category, 3 distinct): ['N', 'IE', 'EI']
====
Features:

position_-30 (category, 5 distinct): ['G', 'C', 'A', 'T', 'D']
position_-29 (category, 5 distinct): ['C', 'G', 'A', 'T', 'D']
position_-28 (category, 4 distinct): ['C', 'G', 'T', 'A']
position_-27 (category, 4 distinct): ['C', 'G', 'A', 'T']
position_-26 (category, 4 distinct): ['C', 'T', 'A', 'G']
position_-25 (category, 4 distinct): ['C', 'G', 'T', 'A']
position_-24 (category, 4 distinct): ['C', 'A', 'G', 'T']
position_-23 (category, 4 distinct): ['C', 'T', 'A', 'G']
position_-22 (category, 4 distinct): ['C', 'T', 'A', 'G']
position_-21 (category, 4 distinct): ['T', 'C', 'A', 'G']
position_-20 (category, 4 distinct): ['T', 'C', 'G', 'A']
position_-19 (category, 4 distinct): ['C', 'G', 'T', 'A']
position_-18 (category, 4 distinct): ['C', 'T', 'G', 'A']
position_-17 (category, 5 distinct): ['C', 'A', 'T', 'G', 'N']
position_-16 (category, 4 distinct): ['C', 'T', 'G', 'A']
position_-15 (category, 4 distinct): ['C', 'T', 'A', 'G']
position_-14 (category, 4 distinct): ['T', 'C', 'G', 'A']
position_-13 (category, 4 distinct): ['T', 'C', 'G', 'A']
position_-12 (category, 5 distinct): ['C', 'T', 'G', 'A', 'N']
position_-11 (category, 5 distinct): ['T', 'C', 'A', 'G', 'N']
position_-10 (category, 5 distinct): ['C', 'T', 'G', 'A', 'N']
position_-9 (category, 5 distinct): ['C', 'T', 'G', 'A', 'N']
position_-8 (category, 5 distinct): ['C', 'T', 'G', 'A', 'N']
position_-7 (category, 5 distinct): ['C', 'T', 'G', 'A', 'N']
position_-6 (category, 5 distinct): ['C', 'T', 'G', 'A', 'N']
position_-5 (category, 5 distinct): ['T', 'C', 'A', 'G', 'N']
position_-4 (category, 5 distinct): ['C', 'G', 'A', 'T', 'N']
position_-3 (category, 5 distinct): ['C', 'A', 'T', 'G', 'N']
position_-2 (category, 5 distinct): ['A', 'C', 'G', 'T', 'N']
position_-1 (category, 5 distinct): ['G', 'A', 'T', 'C', 'N']
position_1 (category, 5 distinct): ['G', 'A', 'C', 'T', 'N']
position_2 (category, 5 distinct): ['T', 'A', 'C', 'G', 'N']
position_3 (category, 5 distinct): ['A', 'G', 'C', 'T', 'N']
position_4 (category, 5 distinct): ['A', 'G', 'C', 'T', 'N']
position_5 (category, 6 distinct): ['G', 'C', 'T', 'A', 'N', 'R']
position_6 (category, 6 distinct): ['T', 'C', 'G', 'A', 'N', 'S']
position_7 (category, 5 distinct): ['G', 'A', 'C', 'T', 'N']
position_8 (category, 5 distinct): ['C', 'G', 'A', 'T', 'N']
position_9 (category, 5 distinct): ['C', 'G', 'T', 'A', 'N']
position_10 (category, 5 distinct): ['G', 'C', 'T', 'A', 'N']
position_11 (category, 5 distinct): ['G', 'C', 'T', 'A', 'N']
position_12 (category, 5 distinct): ['G', 'C', 'T', 'A', 'N']
position_13 (category, 5 distinct): ['G', 'C', 'A', 'T', 'N']
position_14 (category, 5 distinct): ['C', 'G', 'T', 'A', 'N']
position_15 (category, 5 distinct): ['C', 'G', 'A', 'T', 'N']
position_16 (category, 5 distinct): ['G', 'C', 'A', 'T', 'N']
position_17 (category, 5 distinct): ['G', 'T', 'C', 'A', 'N']
position_18 (category, 5 distinct): ['G', 'C', 'A', 'T', 'N']
position_19 (category, 5 distinct): ['G', 'C', 'A', 'T', 'N']
position_20 (category, 5 distinct): ['G', 'C', 'A', 'T', 'N']
position_21 (category, 5 distinct): ['G', 'C', 'T', 'A', 'N']
position_22 (category, 5 distinct): ['G', 'A', 'C', 'T', 'N']
position_23 (category, 5 distinct): ['G', 'C', 'T', 'A', 'N']
position_24 (category, 5 distinct): ['G', 'C', 'T', 'A', 'N']
position_25 (category, 5 distinct): ['G', 'C', 'A', 'T', 'N']
position_26 (category, 5 distinct): ['G', 'C', 'T', 'A', 'N']
position_27 (category, 5 distinct): ['G', 'C', 'T', 'A', 'N']
position_28 (category, 5 distinct): ['C', 'G', 'T', 'A', 'N']
position_29 (category, 5 distinct): ['C', 'G', 'A', 'T', 'N']
position_30 (category, 5 distinct): ['G', 'C', 'T', 'A', 'N']
'''

CONTEXT = "Genetics primate splice-junction gene sequences (DNA)"
TARGET = CuratedTarget(raw_name="SiteType", task_type=SupervisedTask.MULTICLASS)