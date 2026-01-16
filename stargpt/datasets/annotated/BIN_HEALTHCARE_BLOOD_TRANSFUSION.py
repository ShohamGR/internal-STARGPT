from tabstar_paper.datasets.curation_objects import CuratedTarget
from tabstar_paper.datasets.objects import SupervisedTask

'''
Dataset Name: blood-transfusion-service-center
====
Examples: 748
====
URL: https://www.openml.org/search?type=data&id=46913
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
- **Original Data Source:** https://doi.org/10.24432/C5GS39
- **Reference (please cite)**: Yeh, I-Cheng, King-Jang Yang, and Tao-Ming Ting. 'Knowledge discovery on RFM model using Bernoulli sequence.' Expert Systems with applications 36.3 (2009): 5866-5871. https://doi.org/10.1016/j.eswa.2008.07.018
- **Dataset Year:** 2008
- **Dataset Description:** see the reference and the original data source for details.

#### Curation comments by the TabArena team (for code see the [page of the study](https://tabarena.ai/data-tabular-ml-iid-study)):
- We made feature names more descriptive.
- Anomaly: the data has a lot of duplicates (29%) and several duplicates with 
  different target values.
  
====
Description for 1464: 

**Author**: Prof. I-Cheng Yeh  
**Source**: [UCI](https://archive.ics.uci.edu/ml/datasets/Blood+Transfusion+Service+Center)  
**Please cite**: Yeh, I-Cheng, Yang, King-Jang, and Ting, Tao-Ming, "Knowledge discovery on RFM model using Bernoulli sequence", Expert Systems with Applications, 2008.   

**Blood Transfusion Service Center Data Set**  
Data taken from the Blood Transfusion Service Center in Hsin-Chu City in Taiwan -- this is a classification problem.

To demonstrate the RFMTC marketing model (a modified version of RFM), this study adopted the donor database of Blood Transfusion Service Center in Hsin-Chu City in Taiwan. The center passes their blood transfusion service bus to one university in Hsin-Chu City to gather blood donated about every three months. To build an FRMTC model, we selected 748 donors at random from the donor database. 

### Attribute Information  
* V1: Recency - months since last donation
* V2: Frequency - total number of donation
* V3: Monetary - total blood donated in c.c.
* V4: Time - months since first donation), and a binary variable representing whether he/she donated blood in March 2007 (1 stand for donating blood; 0 stands for not donating blood).

The target attribute is a binary variable representing whether he/she donated blood in March 2007 (2 stands for donating blood; 1 stands for not donating blood).

====
Target Variable: DonatedBloodInMarch2007 (category, 2 distinct): ['No', 'Yes']
====
Features:

MonthsSinceLastDonation (uint8, 31 distinct): ['2', '4', '11', '14', '16', '23', '21', '9', '3', '1']
NumberOfDonations (uint8, 33 distinct): ['1', '2', '3', '4', '5', '6', '7', '8', '9', '11']
TotalBloodDonated (int64, 33 distinct): ['250', '500', '750', '1000', '1250', '1500', '1750', '2000', '2250', '2750']
MonthsSinceFirstDonation (uint8, 78 distinct): ['4', '16', '14', '2', '23', '28', '26', '11', '35', '21']
'''

TARGET = CuratedTarget(raw_name="DonatedBloodInMarch2007", task_type=SupervisedTask.BINARY)
