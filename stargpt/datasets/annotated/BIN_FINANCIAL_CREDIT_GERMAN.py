from tabstar_paper.datasets.curation_objects import CuratedFeature, CuratedTarget
from tabstar_paper.datasets.objects import SupervisedTask, FeatureType

'''
Dataset Name: credit-g
====
Examples: 1000
====
URL: https://www.openml.org/search?type=data&id=46918
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
- **Original Data Source:** https://doi.org/10.24432/C5NC77
- **Reference (please cite)**: Hofmann, H. (1994). Statlog (German Credit Data) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77.
- **Dataset Year:** 1994
- **Dataset Description:** see the reference and the original data source for details.

#### Curation comments by the TabArena team (for code see the [page of the study](https://tabarena.ai/data-tabular-ml-iid-study)):
- We revered the original ordinal encoding. 
- Anomaly: the original task used a cost matrix for evaluation.
====

# This is the description of dataset: 31
====
Description: **Author**: Dr. Hans Hofmann  
**Source**: [UCI](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)) - 1994    
**Please cite**: [UCI](https://archive.ics.uci.edu/ml/citation_policy.html)

**German Credit dataset**  
This dataset classifies people described by a set of attributes as good or bad credit risks.

This dataset comes with a cost matrix: 
``` 
Good  Bad (predicted)  
Good   0    1   (actual)  
Bad    5    0  
```

It is worse to class a customer as good when they are bad (5), than it is to class a customer as bad when they are good (1).  

### Attribute description  

1. Status of existing checking account, in Deutsche Mark.  
2. Duration in months  
3. Credit history (credits taken, paid back duly, delays, critical accounts)  
4. Purpose of the credit (car, television,...)  
5. Credit amount  
6. Status of savings account/bonds, in Deutsche Mark.  
7. Present employment, in number of years.  
8. Installment rate in percentage of disposable income  
9. Personal status (married, single,...) and sex  
10. Other debtors / guarantors  
11. Present residence since X years  
12. Property (e.g. real estate)  
13. Age in years  
14. Other installment plans (banks, stores)  
15. Housing (rent, own,...)  
16. Number of existing credits at this bank  
17. Job  
18. Number of people being liable to provide maintenance for  
19. Telephone (yes,no)  
20. Foreign worker (yes,no)
====

Target Variable: good_or_bad_customer (category, 2 distinct): ['good', 'bad']
====
Features:

checking_status (category, 4 distinct): ['no checking account', '<0 DM', '0 <= ... < 200 DM', '>= 200 DM / salary assignments for >= 1 year']
duration_months (uint8, 33 distinct): ['24', '12', '18', '36', '6', '15', '9', '48', '30', '21']
credit_history (category, 5 distinct): ['existing credits paid duly till now', 'critical account / other credits existing', 'delay in paying off in past', 'all credits at this bank paid duly', 'no credits taken / all paid duly']
credit_purpose (category, 10 distinct): ['radio/television', 'car (new)', 'furniture/equipment', 'car (used)', 'business', 'education', 'repairs', 'domestic appliances', 'others', 'retraining']
credit_amount (int64, 921 distinct): ['1275', '1262', '1393', '1478', '1258', '1237', '1413', '4272', '2333', '3832']
savings_status (category, 5 distinct): ['< 100 DM', 'unknown / no savings', '100 <= ... < 500 DM', '500 <= ... < 1000 DM', '>= 1000 DM']
employment_since (category, 5 distinct): ['1 <= ... < 4 years', '>= 7 years', '4 <= ... < 7 years', '< 1 year', 'unemployed']
installment_rate_percent (uint8, 4 distinct): ['4', '2', '3', '1']
personal_status_sex (category, 4 distinct): ['male: single', 'female: divorced/separated/married', 'male: married/widowed', 'male: divorced/separated']
other_debtors (category, 3 distinct): ['none', 'guarantor', 'co-applicant']
residence_since (uint8, 4 distinct): ['4', '2', '3', '1']
property (category, 4 distinct): ['car or other (not savings)', 'real estate', 'building society savings / life insurance', 'unknown / no property']
age_years (uint8, 53 distinct): ['27', '26', '23', '24', '28', '25', '35', '30', '36', '31']
other_installment_plans (category, 3 distinct): ['none', 'bank', 'stores']
housing (category, 3 distinct): ['own', 'rent', 'for free']
existing_credits_count (uint8, 4 distinct): ['1', '2', '3', '4']
job (category, 4 distinct): ['skilled employee / official', 'unskilled resident', 'management / self-employed / highly qualified', 'unemployed / unskilled non-resident']
people_liable (uint8, 2 distinct): ['1', '2']
telephone (category, 2 distinct): ['none', 'yes, registered']
foreign_worker (category, 2 distinct): ['yes', 'no']
'''

CONTEXT = "Financial credit history of German customers"
TARGET = CuratedTarget(raw_name="good_or_bad_customer", task_type=SupervisedTask.BINARY)

