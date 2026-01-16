from tabstar_paper.datasets.curation_objects import CuratedTarget
from tabstar_paper.datasets.objects import SupervisedTask

'''
Dataset Name: website_phishing
====
Examples: 1353
====
URL: https://www.openml.org/search?type=data&id=46963
====
Description: This dataset was curated for [TabArena](https://tabarena.ai/) by the TabArena team
as part of the [TabArena Tabular ML IID Study](https://tabarena.ai/data-tabular-ml-iid-study).
For more details on the study, see our [paper](https://tabarena.ai/paper-tabular-ml-iid-study).

**Dataset Focus**: This dataset shall be used for evaluating predictive machine
learning models for independent and identically distributed tabular data. The
intended task is classification.

====
Description of 4534

**Author**: Rami Mustafa A Mohammad ( University of Huddersfield","rami.mohammad '@' hud.ac.uk","rami.mustafa.a '@' gmail.com) Lee McCluskey (University of Huddersfield","t.l.mccluskey '@' hud.ac.uk )  Fadi Thabtah (Canadian University of Dubai","fadi '@' cud.ac.ae)  
**Source**: [UCI](https://archive.ics.uci.edu/ml/datasets/phishing+websites)  
**Please cite**: Please refer to the [Machine Learning Repository's citation policy](https://archive.ics.uci.edu/ml/citation_policy.html)  

Source:

Rami Mustafa A Mohammad ( University of Huddersfield, rami.mohammad '@' hud.ac.uk, rami.mustafa.a '@' gmail.com)
Lee McCluskey (University of Huddersfield,t.l.mccluskey '@' hud.ac.uk )
Fadi Thabtah (Canadian University of Dubai,fadi '@' cud.ac.ae)


Data Set Information:

One of the challenges faced by our research was the unavailability of reliable training datasets. In fact this challenge faces any researcher in the field. However, although plenty of articles about predicting phishing websites have been disseminated these days, no reliable training dataset has been published publically, may be because there is no agreement in literature on the definitive features that characterize phishing webpages, hence it is difficult to shape a dataset that covers all possible features. 
In this dataset, we shed light on the important features that have proved to be sound and effective in predicting phishing websites. In addition, we propose some new features.


Attribute Information:

For Further information about the features see the features file in the [data folder](https://archive.ics.uci.edu/ml/machine-learning-databases/00327/Phishing Websites Features.docx) of UCI.

Relevant Papers:

Mohammad, Rami, McCluskey, T.L. and Thabtah, Fadi (2012) An Assessment of Features Related to Phishing Websites using an Automated Technique. In: International Conferece For Internet Technology And Secured Transactions. ICITST 2012 . IEEE, London, UK, pp. 492-497. ISBN 978-1-4673-5325-0

Mohammad, Rami, Thabtah, Fadi Abdeljaber and McCluskey, T.L. (2014) Predicting phishing websites based on self-structuring neural network. Neural Computing and Applications, 25 (2). pp. 443-458. ISSN 0941-0643

Mohammad, Rami, McCluskey, T.L. and Thabtah, Fadi Abdeljaber (2014) Intelligent Rule based Phishing Websites Classification. IET Information Security, 8 (3). pp. 153-160. ISSN 1751-8709

 

Citation Request:

Please refer to the Machine Learning Repository's citation policy
====

---
#### Dataset Metadata
- **Licence:** CC BY 4.0
- **Original Data Source:** https://doi.org/10.24432/C5B301
- **Reference (please cite)**: Abdelhamid, Neda, Aladdin Ayesh, and Fadi Thabtah. 'Phishing detection based associative classification data mining.' Expert Systems with Applications 41.13 (2014): 5948-5959. https://doi.org/10.1016/j.eswa.2014.03.019
- **Dataset Year:** 2014
- **Dataset Description:** see the reference and the original data source for details.

#### Curation comments by the TabArena team (for code see the [page of the study](https://tabarena.ai/data-tabular-ml-iid-study)):
- We reversed the ordinal encoding of the original data.
- We renamed the target feature to be more meaningful.
- Anomaly: the data has many duplicates (~50%).
====
Target Variable: WebsiteType (category, 3 distinct): ['Phishy', 'Legitimate', 'Suspicious']
====
Features:

SFH (category, 3 distinct): ['Legitimate', 'Phishy', 'Suspicious']
popUpWidnow (category, 3 distinct): ['Suspicious', 'Phishy', 'Legitimate']
SSLfinal_State (category, 3 distinct): ['Legitimate', 'Phishy', 'Suspicious']
Request_URL (category, 3 distinct): ['Phishy', 'Suspicious', 'Legitimate']
URL_of_Anchor (category, 3 distinct): ['Phishy', 'Legitimate', 'Suspicious']
web_traffic (category, 3 distinct): ['Suspicious', 'Legitimate', 'Phishy']
URL_Length (category, 3 distinct): ['Suspicious', 'Phishy', 'Legitimate']
age_of_domain (category, 2 distinct): ['Legitimate', 'Phishy']
having_IP_Address (category, 2 distinct): ['Suspicious', 'Legitimate']
'''

CONTEXT = "Phishing Detection Websites from Huddersfield University"
TARGET = CuratedTarget(raw_name="WebsiteType", task_type=SupervisedTask.MULTICLASS)
