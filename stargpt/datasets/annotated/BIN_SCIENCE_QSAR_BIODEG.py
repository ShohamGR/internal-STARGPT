from tabstar_paper.datasets.curation_objects import CuratedTarget
from tabstar_paper.datasets.objects import SupervisedTask

'''
Dataset Name: qsar-biodeg
====
Examples: 1054
====
URL: https://www.openml.org/search?type=data&id=46952
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
- **Original Data Source:** https://doi.org/10.24432/C5H60M
- **Reference (please cite)**: Mansouri, Kamel, et al. 'Quantitative structureactivity relationship models for ready biodegradability of chemicals.' Journal of chemical information and modeling 53.4 (2013): 867-878. https://doi.org/10.1021/ci4000213
- **Dataset Year:** 2013
- **Dataset Description:** see the reference and the original data source for details.

#### Curation comments by the TabArena team (for code see the [page of the study](https://tabarena.ai/data-tabular-ml-iid-study)):
- We added semantic meaningful feature names.
- Anomaly: several features are numeric-ordinal in nature but it is unclear if they are categorical features.
====

# This is the description of dataset: 1494

====
Description: **Author**: Kamel Mansouri, Tine Ringsted, Davide Ballabio  
**Source**: [UCI](https://archive.ics.uci.edu/ml/datasets/QSAR+biodegradation)  
**Please cite**: Mansouri, K., Ringsted, T., Ballabio, D., Todeschini, R., Consonni, V. (2013). Quantitative Structure - Activity Relationship models for ready biodegradability of chemicals. Journal of Chemical Information and Modeling, 53, 867-878 


QSAR biodegradation Data Set 

* Abstract: 

Data set containing values for 41 attributes (molecular descriptors) used to classify 1055 chemicals into 2 classes (ready and not ready biodegradable).


* Source:

Kamel Mansouri, Tine Ringsted, Davide Ballabio (davide.ballabio '@' unimib.it), Roberto Todeschini, Viviana Consonni, Milano Chemometrics and QSAR Research Group (http://michem.disat.unimib.it/chm/), UniversitÃ  degli Studi Milano â€“ Bicocca, Milano (Italy)


* Data Set Information:

The QSAR biodegradation dataset was built in the Milano Chemometrics and QSAR Research Group (UniversitÃ  degli Studi Milano â€“ Bicocca, Milano, Italy). The research leading to these results has received funding from the European Communityâ€™s Seventh Framework Programme [FP7/2007-2013] under Grant Agreement n. 238701 of Marie Curie ITN Environmental Chemoinformatics (ECO) project. 
The data have been used to develop QSAR (Quantitative Structure Activity Relationships) models for the study of the relationships between chemical structure and biodegradation of molecules. Biodegradation experimental values of 1055 chemicals were collected from the webpage of the National Institute of Technology and Evaluation of Japan (NITE). Classification models were developed in order to discriminate ready (356) and not ready (699) biodegradable molecules by means of three different modelling methods: k Nearest Neighbours, Partial Least Squares Discriminant Analysis and Support Vector Machines. Details on attributes (molecular descriptors) selected in each model can be found in the quoted reference: Mansouri, K., Ringsted, T., Ballabio, D., Todeschini, R., Consonni, V. (2013). Quantitative Structure - Activity Relationship models for ready biodegradability of chemicals. Journal of Chemical Information and Modeling, 53, 867-878.


* Attribute Information:

41 molecular descriptors and 1 experimental class: 
1) SpMax_L: Leading eigenvalue from Laplace matrix 
2) J_Dz(e): Balaban-like index from Barysz matrix weighted by Sanderson electronegativity 
3) nHM: Number of heavy atoms 
4) F01[N-N]: Frequency of N-N at topological distance 1 
5) F04[C-N]: Frequency of C-N at topological distance 4 
6) NssssC: Number of atoms of type ssssC 
7) nCb-: Number of substituted benzene C(sp2) 
8) C%: Percentage of C atoms 
9) nCp: Number of terminal primary C(sp3) 
10) nO: Number of oxygen atoms 
11) F03[C-N]: Frequency of C-N at topological distance 3 
12) SdssC: Sum of dssC E-states 
13) HyWi_B(m): Hyper-Wiener-like index (log function) from Burden matrix weighted by mass 
14) LOC: Lopping centric index 
15) SM6_L: Spectral moment of order 6 from Laplace matrix 
16) F03[C-O]: Frequency of C - O at topological distance 3 
17) Me: Mean atomic Sanderson electronegativity (scaled on Carbon atom) 
18) Mi: Mean first ionization potential (scaled on Carbon atom) 
19) nN-N: Number of N hydrazines 
20) nArNO2: Number of nitro groups (aromatic) 
21) nCRX3: Number of CRX3 
22) SpPosA_B(p): Normalized spectral positive sum from Burden matrix weighted by polarizability 
23) nCIR: Number of circuits 
24) B01[C-Br]: Presence/absence of C - Br at topological distance 1 
25) B03[C-Cl]: Presence/absence of C - Cl at topological distance 3 
26) N-073: Ar2NH / Ar3N / Ar2N-Al / R..N..R 
27) SpMax_A: Leading eigenvalue from adjacency matrix (Lovasz-Pelikan index) 
28) Psi_i_1d: Intrinsic state pseudoconnectivity index - type 1d 
29) B04[C-Br]: Presence/absence of C - Br at topological distance 4 
30) SdO: Sum of dO E-states 
31) TI2_L: Second Mohar index from Laplace matrix 
32) nCrt: Number of ring tertiary C(sp3) 
33) C-026: R--CX--R 
34) F02[C-N]: Frequency of C - N at topological distance 2 
35) nHDon: Number of donor atoms for H-bonds (N and O) 
36) SpMax_B(m): Leading eigenvalue from Burden matrix weighted by mass 
37) Psi_i_A: Intrinsic state pseudoconnectivity index - type S average 
38) nN: Number of Nitrogen atoms 
39) SM6_B(m): Spectral moment of order 6 from Burden matrix weighted by mass 
40) nArCOOR: Number of esters (aromatic) 
41) nX: Number of halogen atoms 
42) experimental class: ready biodegradable (RB) and not ready biodegradable (NRB)


* Relevant Papers:

Mansouri, K., Ringsted, T., Ballabio, D., Todeschini, R., Consonni, V. (2013). Quantitative Structure - Activity Relationship models for ready biodegradability of chemicals. Journal of Chemical Information and Modeling, 53, 867-878
====
Target Variable: Biodegradable (category, 2 distinct): ['No', 'Yes']
====
Features:

Laplace_Leading_Eigenvalue (float64, 440 distinct): ['4.414', '4.732', '4.0', '4.17', '4.562', '4.303', '4.807', '4.77', '4.499', '4.214']
Weighted_Balaban_Index_Barysz_Matrix (float64, 1021 distinct): ['2.0', '3.0864', '3.3983', '2.7059', '3.2871', '3.6943', '1.8789', '2.4524', '3.0785', '3.4572']
Num_Heavy_Atoms (uint8, 11 distinct): ['0', '1', '2', '3', '4', '6', '5', '7', '8', '10']
Freq_NN_At_Dist1 (uint8, 4 distinct): ['0', '1', '2', '3']
Freq_CN_At_Dist4 (uint8, 16 distinct): ['0', '2', '1', '3', '4', '6', '7', '5', '8', '9']
Num_ssssC_Atoms (uint8, 13 distinct): ['0', '1', '2', '3', '4', '6', '5', '9', '8', '11']
Num_Substituted_BenzeneC (uint8, 15 distinct): ['0', '2', '3', '4', '1', '6', '5', '8', '7', '12']
Percentage_C_Atoms (float64, 188 distinct): ['33.3', '40.0', '50.0', '25.0', '42.9', '28.6', '30.0', '37.5', '46.2', '41.2']
Num_Terminal_PrimaryC (uint8, 15 distinct): ['0', '1', '2', '3', '4', '6', '5', '8', '9', '7']
Num_Oxygen_Atoms (uint8, 12 distinct): ['0', '2', '1', '4', '3', '6', '5', '7', '8', '12']
Freq_CN_At_Dist3 (uint8, 21 distinct): ['0', '2', '4', '1', '3', '6', '8', '5', '12', '10']
Sum_dssC_EStates (float64, 384 distinct): ['0.0', '-1.093', '-2.514', '-1.072', '-0.98', '-0.888', '0.134', '-0.875', '-0.664', '-0.945']
Weighted_HyperWiener_Index_Burden_Matrix (float64, 755 distinct): ['3.647', '3.66', '3.192', '3.462', '3.375', '3.233', '3.37', '3.772', '3.699', '4.049']
Lopping_Centric_Index (float64, 373 distinct): ['0.0', '0.875', '1.185', '0.881', '1.187', '1.16', '0.971', '1.459', '0.918', '0.802']
Laplace_Spectral_Moment6 (float64, 510 distinct): ['9.54', '8.597', '9.882', '9.833', '9.183', '9.311', '10.099', '8.755', '9.863', '7.408']
Freq_CO_At_Dist3 (uint8, 24 distinct): ['0', '2', '4', '6', '8', '1', '3', '12', '9', '10']
Mean_Sanderson_Electronegativity (float64, 167 distinct): ['0.998', '0.979', '0.993', '0.974', '0.987', '0.991', '1.014', '0.983', '0.98', '1.02']
Mean_Ionization_Potential (float64, 125 distinct): ['1.127', '1.139', '1.14', '1.125', '1.129', '1.146', '1.121', '1.141', '1.144', '1.132']
Num_N_Hydrazine (uint8, 3 distinct): ['0', '1', '2']
Num_Aromatic_Nitro_Groups (uint8, 4 distinct): ['0', '1', '2', '3']
Num_CRX3 (uint8, 4 distinct): ['0', '1', '2', '3']
Weighted_Normalized_SpectralPositiveSum_Burden_Matrix (float64, 352 distinct): ['1.195', '1.299', '1.254', '1.296', '1.253', '1.28', '1.211', '1.295', '1.215', '1.202']
Num_Circuits (uint8, 13 distinct): ['1', '0', '2', '3', '6', '4', '7', '15', '5', '10']
Presence_CBr_At_Dist1 (category, 2 distinct): ['0', '1']
Presence_CCl_At_Dist3 (category, 2 distinct): ['0', '1']
N073_chemical_substructure (category, 4 distinct): ['0', '1', '2', '3']
Adjacency_LeadingEigenvalue (float64, 329 distinct): ['2.0', '2.236', '2.194', '1.848', '2.175', '2.303', '2.101', '1.732', '1.902', '2.136']
Intrinsic_State_Pseudoconnectivity (float64, 204 distinct): ['0.0', '-0.008', '0.001', '0.004', '-0.002', '-0.001', '0.014', '0.015', '-0.025', '-0.007']
Presence_CBr_At_Dist4 (category, 2 distinct): ['0', '1']
Sum_dO_EStates (float64, 470 distinct): ['0.0', '19.107', '10.143', '22.204', '10.87', '10.118', '11.115', '20.135', '10.582', '9.431']
Laplace_MoharIndex2 (float64, 553 distinct): ['1.542', '0.975', '0.95', '1.06', '1.74', '1.481', '1.0', '1.14', '2.052', '1.707']
Num_RingTertiaryC (uint8, 8 distinct): ['0', '1', '2', '4', '6', '3', '8', '5']
C026_chemical_substructure (category, 11 distinct): ['0', '1', '2', '3', '4', '6', '5', '10', '12', '8']
Freq_CN_At_Dist2 (uint8, 16 distinct): ['0', '2', '4', '3', '1', '6', '8', '5', '10', '18']
Num_HBond_Donors_Atoms (uint8, 8 distinct): ['0', '1', '2', '3', '4', '6', '5', '7']
Weighted_LeadingEigenvalue_Burden_Matrix (float64, 704 distinct): ['4.009', '6.88', '3.309', '3.423', '3.497', '3.834', '3.712', '3.728', '3.794', '3.876']
Intrinsic_State_Pseudoconnectivity_SAvg (float64, 623 distinct): ['2.833', '2.167', '2.5', '2.667', '1.833', '2.0', '2.333', '2.802', '2.25', '2.611']
Num_Nitrogen_Atoms (uint8, 8 distinct): ['0', '1', '2', '3', '4', '6', '5', '8']
Weighted_SpectralMoment6_Burden_Matrix (float64, 861 distinct): ['8.497', '8.68', '8.506', '8.704', '8.562', '8.015', '8.128', '8.143', '8.601', '9.118']
Num_Esters (uint8, 5 distinct): ['0', '2', '1', '4', '3']
Num_Halogen_Atoms (uint8, 17 distinct): ['0', '1', '2', '3', '4', '6', '5', '10', '8', '7']
'''

CONTEXT = "Quantitative Structure Activity Relationships (QSAR) Biodegradation"
TARGET = CuratedTarget(raw_name="Biodegradable", task_type=SupervisedTask.BINARY)
