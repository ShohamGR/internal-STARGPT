from tabstar_paper.datasets.curation_objects import CuratedTarget
from tabstar_paper.datasets.objects import SupervisedTask

'''
Dataset Name: MIC
====
Examples: 1699
====
URL: https://www.openml.org/search?type=data&id=46980
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
- **Original Data Source:** https://doi.org/10.24432/C53P5M
- **Reference (please cite)**: Golovenkin, Sergey E., et al. 'Trajectories, bifurcations, and pseudo-time in large clinical datasets: applications to myocardial infarction and diabetes data.' GigaScience 9.11 (2020): giaa128. https://doi.org/10.1093/gigascience/giaa128
- **Dataset Year:** 2020
- **Dataset Description:** see the reference and the original data source for details.

#### Curation comments by the TabArena team (for code see the [page of the study](https://tabarena.ai/data-tabular-ml-iid-study)):
- There are 12 possible targets and four possible time moments to predict the targets for this dataset. We use the categorical target as it summarizes the other possible targets. Furthermore, we use the last possible time point for prediction to utilize all available data. 
- We treat "?" as missing values.
- We dropped the "id" column.
- We reversed the ordinal encoding of the target feature.
====
Target Variable: LET_IS (category, 8 distinct): ['alive', 'cardiogenic_shock', 'myocardial_rupture', 'asystole', 'ventricular_fibrillation', 'progress_congestive_heart_failure', 'pulmonary_edema', 'thromboembolism']
====
Features:

AGE (float64, 62 distinct): ['63.0', '65.0', '62.0', '64.0', '70.0', '52.0', '66.0', '61.0', '55.0', '57.0']
SEX (category, 2 distinct): ['1', '0']
INF_ANAM (category, 5 distinct): ['0.0', '1.0', '2.0', '3.0', 'nan']
STENOK_AN (category, 8 distinct): ['0.0', '6.0', '1.0', '2.0', '5.0', '3.0', 'nan', '4.0']
FK_STENOK (category, 6 distinct): ['2.0', '0.0', 'nan', '3.0', '1.0', '4.0']
IBS_POST (category, 4 distinct): ['2.0', '1.0', '0.0', 'nan']
IBS_NASL (category, 3 distinct): ['nan', '0.0', '1.0']
GB (category, 5 distinct): ['2.0', '0.0', '3.0', '1.0', 'nan']
SIM_GIPERT (category, 3 distinct): ['0.0', '1.0', 'nan']
DLIT_AG (category, 9 distinct): ['0.0', '7.0', 'nan', '6.0', '1.0', '5.0', '2.0', '3.0', '4.0']
ZSN_A (category, 6 distinct): ['0.0', '1.0', 'nan', '3.0', '2.0', '4.0']
nr_11 (category, 3 distinct): ['0.0', '1.0', 'nan']
nr_01 (category, 3 distinct): ['0.0', 'nan', '1.0']
nr_02 (category, 3 distinct): ['0.0', 'nan', '1.0']
nr_03 (category, 3 distinct): ['0.0', '1.0', 'nan']
nr_04 (category, 3 distinct): ['0.0', '1.0', 'nan']
nr_07 (category, 3 distinct): ['0.0', 'nan', '1.0']
nr_08 (category, 3 distinct): ['0.0', 'nan', '1.0']
np_01 (category, 3 distinct): ['0.0', 'nan', '1.0']
np_04 (category, 3 distinct): ['0.0', 'nan', '1.0']
np_05 (category, 3 distinct): ['0.0', 'nan', '1.0']
np_07 (category, 3 distinct): ['0.0', 'nan', '1.0']
np_08 (category, 3 distinct): ['0.0', 'nan', '1.0']
np_09 (category, 3 distinct): ['0.0', 'nan', '1.0']
np_10 (category, 3 distinct): ['0.0', 'nan', '1.0']
endocr_01 (category, 3 distinct): ['0.0', '1.0', 'nan']
endocr_02 (category, 3 distinct): ['0.0', '1.0', 'nan']
endocr_03 (category, 3 distinct): ['0.0', '1.0', 'nan']
zab_leg_01 (category, 3 distinct): ['0.0', '1.0', 'nan']
zab_leg_02 (category, 3 distinct): ['0.0', '1.0', 'nan']
zab_leg_03 (category, 3 distinct): ['0.0', '1.0', 'nan']
zab_leg_04 (category, 3 distinct): ['0.0', '1.0', 'nan']
zab_leg_06 (category, 3 distinct): ['0.0', '1.0', 'nan']
S_AD_KBRIG (float64, 30 distinct): ['140.0', '130.0', '160.0', '120.0', '110.0', '150.0', '170.0', '180.0', '100.0', '80.0']
D_AD_KBRIG (float64, 21 distinct): ['80.0', '90.0', '100.0', '70.0', '60.0', '120.0', '110.0', '40.0', '0.0', '85.0']
S_AD_ORIT (float64, 32 distinct): ['130.0', '120.0', '140.0', '160.0', '110.0', '150.0', '180.0', '100.0', '170.0', '90.0']
D_AD_ORIT (float64, 20 distinct): ['80.0', '90.0', '100.0', '70.0', '60.0', '110.0', '40.0', '120.0', '0.0', '50.0']
O_L_POST (category, 3 distinct): ['0.0', '1.0', 'nan']
K_SH_POST (category, 3 distinct): ['0.0', '1.0', 'nan']
MP_TP_POST (category, 3 distinct): ['0.0', '1.0', 'nan']
SVT_POST (category, 3 distinct): ['0.0', 'nan', '1.0']
GT_POST (category, 3 distinct): ['0.0', 'nan', '1.0']
FIB_G_POST (category, 3 distinct): ['0.0', '1.0', 'nan']
ant_im (category, 6 distinct): ['0.0', '4.0', '1.0', 'nan', '2.0', '3.0']
lat_im (category, 6 distinct): ['1.0', '0.0', '2.0', 'nan', '3.0', '4.0']
inf_im (category, 6 distinct): ['0.0', '1.0', '2.0', '4.0', '3.0', 'nan']
post_im (category, 6 distinct): ['0.0', '1.0', 'nan', '2.0', '3.0', '4.0']
IM_PG_P (category, 3 distinct): ['0.0', '1.0', 'nan']
ritm_ecg_p_01 (category, 3 distinct): ['1.0', '0.0', 'nan']
ritm_ecg_p_02 (category, 3 distinct): ['0.0', 'nan', '1.0']
ritm_ecg_p_04 (category, 3 distinct): ['0.0', 'nan', '1.0']
ritm_ecg_p_06 (category, 3 distinct): ['0.0', 'nan', '1.0']
ritm_ecg_p_07 (category, 3 distinct): ['0.0', '1.0', 'nan']
ritm_ecg_p_08 (category, 3 distinct): ['0.0', 'nan', '1.0']
n_r_ecg_p_01 (category, 3 distinct): ['0.0', 'nan', '1.0']
n_r_ecg_p_02 (category, 3 distinct): ['0.0', 'nan', '1.0']
n_r_ecg_p_03 (category, 3 distinct): ['0.0', '1.0', 'nan']
n_r_ecg_p_04 (category, 3 distinct): ['0.0', 'nan', '1.0']
n_r_ecg_p_05 (category, 3 distinct): ['0.0', 'nan', '1.0']
n_r_ecg_p_06 (category, 3 distinct): ['0.0', 'nan', '1.0']
n_r_ecg_p_08 (category, 3 distinct): ['0.0', 'nan', '1.0']
n_r_ecg_p_09 (category, 3 distinct): ['0.0', 'nan', '1.0']
n_r_ecg_p_10 (category, 3 distinct): ['0.0', 'nan', '1.0']
n_p_ecg_p_01 (category, 3 distinct): ['0.0', 'nan', '1.0']
n_p_ecg_p_03 (category, 3 distinct): ['0.0', 'nan', '1.0']
n_p_ecg_p_04 (category, 3 distinct): ['0.0', 'nan', '1.0']
n_p_ecg_p_05 (category, 3 distinct): ['0.0', 'nan', '1.0']
n_p_ecg_p_06 (category, 3 distinct): ['0.0', 'nan', '1.0']
n_p_ecg_p_07 (category, 3 distinct): ['0.0', 'nan', '1.0']
n_p_ecg_p_08 (category, 3 distinct): ['0.0', 'nan', '1.0']
n_p_ecg_p_09 (category, 3 distinct): ['0.0', 'nan', '1.0']
n_p_ecg_p_10 (category, 3 distinct): ['0.0', 'nan', '1.0']
n_p_ecg_p_11 (category, 3 distinct): ['0.0', 'nan', '1.0']
n_p_ecg_p_12 (category, 3 distinct): ['0.0', 'nan', '1.0']
fibr_ter_01 (category, 3 distinct): ['0.0', '1.0', 'nan']
fibr_ter_02 (category, 3 distinct): ['0.0', '1.0', 'nan']
fibr_ter_03 (category, 3 distinct): ['0.0', '1.0', 'nan']
fibr_ter_05 (category, 3 distinct): ['0.0', 'nan', '1.0']
fibr_ter_06 (category, 3 distinct): ['0.0', 'nan', '1.0']
fibr_ter_07 (category, 3 distinct): ['0.0', 'nan', '1.0']
fibr_ter_08 (category, 3 distinct): ['0.0', 'nan', '1.0']
GIPO_K (category, 3 distinct): ['0.0', '1.0', 'nan']
K_BLOOD (float64, 51 distinct): ['4.0', '3.8', '4.2', '3.9', '3.5', '4.5', '4.1', '4.3', '3.6', '3.7']
GIPER_NA (category, 3 distinct): ['0.0', 'nan', '1.0']
NA_BLOOD (float64, 40 distinct): ['136.0', '140.0', '130.0', '133.0', '138.0', '143.0', '146.0', '134.0', '132.0', '139.0']
ALT_BLOOD (float64, 69 distinct): ['0.15', '0.3', '0.45', '0.23', '0.38', '0.61', '0.75', '0.52', '0.9', '0.68']
AST_BLOOD (float64, 58 distinct): ['0.15', '0.22', '0.07', '0.3', '0.11', '0.18', '0.37', '0.45', '0.26', '0.52']
KFK_BLOOD (float64, 4 distinct): ['1.2', '1.8', '1.4', '3.6']
L_BLOOD (float64, 174 distinct): ['6.9', '7.0', '7.4', '8.0', '6.8', '7.2', '9.0', '7.7', '7.5', '6.0']
ROE (float64, 58 distinct): ['5.0', '3.0', '10.0', '4.0', '7.0', '8.0', '6.0', '12.0', '15.0', '20.0']
TIME_B_S (category, 10 distinct): ['2.0', '9.0', '1.0', '3.0', '6.0', '7.0', 'nan', '8.0', '5.0', '4.0']
R_AB_1_n (category, 5 distinct): ['0.0', '1.0', '2.0', '3.0', 'nan']
R_AB_2_n (category, 5 distinct): ['0.0', '1.0', 'nan', '2.0', '3.0']
R_AB_3_n (category, 5 distinct): ['0.0', 'nan', '1.0', '2.0', '3.0']
NA_KB (category, 3 distinct): ['nan', '1.0', '0.0']
NOT_NA_KB (category, 3 distinct): ['1.0', 'nan', '0.0']
LID_KB (category, 3 distinct): ['nan', '0.0', '1.0']
NITR_S (category, 3 distinct): ['0.0', '1.0', 'nan']
NA_R_1_n (float64, 5 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0']
NA_R_2_n (float64, 4 distinct): ['0.0', '1.0', '2.0', '3.0']
NA_R_3_n (float64, 3 distinct): ['0.0', '1.0', '2.0']
NOT_NA_1_n (category, 6 distinct): ['0.0', '1.0', '2.0', '3.0', 'nan', '4.0']
NOT_NA_2_n (float64, 4 distinct): ['0.0', '1.0', '2.0', '3.0']
NOT_NA_3_n (float64, 3 distinct): ['0.0', '1.0', '2.0']
LID_S_n (category, 3 distinct): ['0.0', '1.0', 'nan']
B_BLOK_S_n (category, 3 distinct): ['0.0', '1.0', 'nan']
ANT_CA_S_n (category, 3 distinct): ['1.0', '0.0', 'nan']
GEPAR_S_n (category, 3 distinct): ['1.0', '0.0', 'nan']
ASP_S_n (category, 3 distinct): ['1.0', '0.0', 'nan']
TIKL_S_n (category, 3 distinct): ['0.0', '1.0', 'nan']
TRENT_S_n (category, 3 distinct): ['0.0', '1.0', 'nan']
'''

CONTEXT = "Myocardial infarction patients"
TARGET = CuratedTarget(raw_name="LET_IS", task_type=SupervisedTask.MULTICLASS)
