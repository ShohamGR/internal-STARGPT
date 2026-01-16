from tabstar_paper.datasets.curation_objects import CuratedTarget
from tabstar_paper.datasets.objects import SupervisedTask

'''
Dataset Name: wine_quality
====
Examples: 6497
====
URL: https://www.openml.org/search?type=data&id=46964
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
- **Original Data Source:** https://doi.org/10.24432/C56S3T
- **Reference (please cite)**: Cortez, Paulo, et al. 'Modeling wine preferences by data mining from physicochemical properties.' Decision support systems 47.4 (2009): 547-553. https://doi.org/10.1016/j.dss.2009.05.016
- **Dataset Year:** 2009
- **Dataset Description:** see the reference and the original data source for details.

#### Curation comments by the TabArena team (for code see the [page of the study](https://tabarena.ai/data-tabular-ml-iid-study)):
- We combine the original datasets for red and white wine into a single dataset with an additional column indicating the type of wine (red or white).
- We treat the task as a regression problem, following the original work and because the target is the median wine quality (of at least 3 evaluations by experts).
- Anomaly: the data has a high number of duplicates (18%).

====

OpenML wine-quality-white  40498
Description: Citation Request:
  This dataset is public available for research. The details are described in [Cortez et al., 2009]. 
  Please include this citation if you plan to use this database:

  P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. 
  Modeling wine preferences by data mining from physicochemical properties.
  In Decision Support Systems, Elsevier, 47(4):547-553. ISSN: 0167-9236.

  Available at: [@Elsevier] http://dx.doi.org/10.1016/j.dss.2009.05.016
                [Pre-press (pdf)] http://www3.dsi.uminho.pt/pcortez/winequality09.pdf
                [bib] http://www3.dsi.uminho.pt/pcortez/dss09.bib

1. Title: Wine Quality 

2. Sources
   Created by: Paulo Cortez (Univ. Minho), Antonio Cerdeira, Fernando Almeida, Telmo Matos and Jose Reis (CVRVV) @ 2009

3. Past Usage:

  P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. 
  Modeling wine preferences by data mining from physicochemical properties.
  In Decision Support Systems, Elsevier, 47(4):547-553. ISSN: 0167-9236.

  In the above reference, two datasets were created, using red and white wine samples.
  The inputs include objective tests (e.g. PH values) and the output is based on sensory data
  (median of at least 3 evaluations made by wine experts). Each expert graded the wine quality 
  between 0 (very bad) and 10 (very excellent). Several data mining methods were applied to model
  these datasets under a regression approach. The support vector machine model achieved the
  best results. Several metrics were computed: MAD, confusion matrix for a fixed error tolerance (T),
  etc. Also, we plot the relative importances of the input variables (as measured by a sensitivity
  analysis procedure).

4. Relevant Information:

   The two datasets are related to red and white variants of the Portuguese &quot;Vinho Verde&quot; wine.
   For more details, consult: http://www.vinhoverde.pt/en/ or the reference [Cortez et al., 2009].
   Due to privacy and logistic issues, only physicochemical (inputs) and sensory (the output) variables 
   are available (e.g. there is no data about grape types, wine brand, wine selling price, etc.).

   These datasets can be viewed as classification or regression tasks.
   The classes are ordered and not balanced (e.g. there are munch more normal wines than
   excellent or poor ones). Outlier detection algorithms could be used to detect the few excellent
   or poor wines. Also, we are not sure if all input variables are relevant. So
   it could be interesting to test feature selection methods. 

5. Number of Instances: red wine - 1599; white wine - 4898. 

6. Number of Attributes: 11 + output attribute

   Note: several of the attributes may be correlated, thus it makes sense to apply some sort of
   feature selection.

7. Attribute information:

   For more information, read [Cortez et al., 2009].

   Input variables (based on physicochemical tests):
   1 - fixed acidity
   2 - volatile acidity
   3 - citric acid
   4 - residual sugar
   5 - chlorides
   6 - free sulfur dioxide
   7 - total sulfur dioxide
   8 - density
   9 - pH
   10 - sulphates
   11 - alcohol
   Output variable (based on sensory data): 
   12 - quality (score between 0 and 10)

8. Missing Attribute Values: None
====
====
Target Variable: median_wine_quality (uint8, 7 distinct): ['6', '5', '7', '4', '8', '3', '9']
====
Features:

fixed_acidity (float64, 106 distinct): ['6.8', '6.6', '6.4', '7.0', '6.9', '7.2', '6.7', '7.1', '6.5', '7.4']
volatile_acidity (float64, 187 distinct): ['0.28', '0.24', '0.26', '0.25', '0.22', '0.27', '0.23', '0.2', '0.3', '0.32']
citric_acid (float64, 89 distinct): ['0.3', '0.28', '0.32', '0.49', '0.26', '0.34', '0.29', '0.27', '0.24', '0.31']
residual_sugar (float64, 316 distinct): ['2.0', '1.8', '1.6', '1.4', '1.2', '2.2', '2.1', '1.9', '1.7', '1.5']
chlorides (float64, 214 distinct): ['0.044', '0.036', '0.042', '0.046', '0.048', '0.05', '0.04', '0.047', '0.045', '0.038']
free_sulfur_dioxide (float64, 135 distinct): ['29.0', '6.0', '26.0', '15.0', '24.0', '31.0', '17.0', '34.0', '35.0', '23.0']
total_sulfur_dioxide (float64, 276 distinct): ['111.0', '113.0', '117.0', '122.0', '124.0', '128.0', '114.0', '98.0', '118.0', '119.0']
density (float64, 998 distinct): ['0.9972', '0.9976', '0.998', '0.992', '0.9928', '0.9986', '0.9962', '0.9966', '0.9956', '0.9968']
pH (float64, 108 distinct): ['3.16', '3.14', '3.22', '3.2', '3.15', '3.19', '3.18', '3.24', '3.12', '3.1']
sulphates (float64, 111 distinct): ['0.5', '0.46', '0.54', '0.44', '0.38', '0.48', '0.52', '0.49', '0.47', '0.45']
alcohol (float64, 111 distinct): ['9.5', '9.4', '9.2', '10.0', '10.5', '11.0', '9.0', '9.8', '10.4', '9.3']
wine_color (category, 2 distinct): ['white', 'red']
'''

CONTEXT = "Wine Quality Estimation for Red and White Wine"
TARGET = CuratedTarget(raw_name="median_wine_quality", task_type=SupervisedTask.REGRESSION)
