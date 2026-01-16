from tabstar_paper.datasets.curation_objects import CuratedTarget
from tabstar_paper.datasets.objects import SupervisedTask

'''
Dataset Name: jm1
====
Examples: 10885
====
URL: https://www.openml.org/search?type=data&id=46979
====
Description: This dataset was curated for [TabArena](https://tabarena.ai/) by the TabArena team
as part of the [TabArena Tabular ML IID Study](https://tabarena.ai/data-tabular-ml-iid-study).
For more details on the study, see our [paper](https://tabarena.ai/paper-tabular-ml-iid-study).

**Dataset Focus**: This dataset shall be used for evaluating predictive machine
learning models for independent and identically distributed tabular data. The
intended task is classification.

====
Description for kc1, dataset 1067, which should be similar: 

**Author**: Mike Chapman, NASA  
**Source**: [tera-PROMISE](http://openscience.us/repo/defect/mccabehalsted/kc1.html) - 2004  
**Please cite**: Sayyad Shirabad, J. and Menzies, T.J. (2005) The PROMISE Repository of Software Engineering Databases. School of Information Technology and Engineering, University of Ottawa, Canada.  
  
**KC1 Software defect prediction**  
One of the NASA Metrics Data Program defect data sets. Data from software for storage management for receiving and processing ground data. Data comes from McCabe and features extractors of source code.  These features were defined in the 70s in an attempt to objectively characterize code features that are associated with software quality.

### Attribute Information  

1. loc             : numeric % McCabe's line count of code
2. v(g)            : numeric % McCabe "cyclomatic complexity"
3. ev(g)           : numeric % McCabe "essential complexity"
4. iv(g)           : numeric % McCabe "design complexity"
5. n               : numeric % total operators + operands
6. v               : numeric % "volume"
7. l               : numeric % "program length"
8. d               : numeric % "difficulty"
9. i               : numeric % "intelligence"
10. e               : numeric % "effort"
11. b               : numeric % 
12. t               : numeric % Halstead's time estimator
13. lOCode          : numeric % Halstead's line count
14. lOComment       : numeric % Halstead's count of lines of comments
15. lOBlank         : numeric % Halstead's count of blank lines
16. lOCodeAndComment: numeric
17. uniq_Op         : numeric % unique operators
18. uniq_Opnd       : numeric % unique operands
19. total_Op        : numeric % total operators
20. total_Opnd      : numeric % total operands
21. branchCount     : numeric % of the flow graph
22. problems        : {false,true} % module has/has not one or more reported defects

### Relevant papers  

- Shepperd, M. and Qinbao Song and Zhongbin Sun and Mair, C. (2013)
Data Quality: Some Comments on the NASA Software Defect Datasets, IEEE Transactions on Software Engineering, 39.

- Tim Menzies and Justin S. Di Stefano (2004) How Good is Your Blind Spot Sampling Policy? 2004 IEEE Conference on High Assurance
Software Engineering.

- T. Menzies and J. DiStefano and A. Orrego and R. Chapman (2004) Assessing Predictors of Software Defects", Workshop on Predictive Software Models, Chicago
====
---
#### Dataset Metadata
- **Licence:** Public Domain
- **Original Data Source:** https://www.openml.org/d/1053
- **Reference (please cite)**: How Good is Your Blind Spot Sampling Policy?; 2003; Tim Menzies and Justin S. Di Stefano; 2004 IEEE Conference on High Assurance Software Engineering (http://menzies.us/pdf/03blind.pdf).
- **Dataset Year:** 2004
- **Dataset Description:** see the reference and the original data source for details.

#### Curation comments by the TabArena team (for code see the [page of the study](https://tabarena.ai/data-tabular-ml-iid-study)):
- We selected this dataset as representative of a set of predictive modeling tasks of software engineering (other examples of such tasks are available on OpenML under the names mozilla4, pc1, pc2, pc3, mc1, kc1, kc2). This dataset was selected as it has the largest sample size.
====
Target Variable: defects (bool, 2 distinct): ['0', '1']
====
Features:

loc (float64, 365 distinct): ['4.0', '5.0', '7.0', '11.0', '17.0', '12.0', '14.0', '8.0', '15.0', '10.0']
v(g) (float64, 108 distinct): ['1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0', '10.0']
ev(g) (float64, 74 distinct): ['1.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0', '10.0', '11.0']
iv(g) (float64, 82 distinct): ['1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0', '10.0']
n (float64, 806 distinct): ['0.0', '4.0', '7.0', '9.0', '5.0', '15.0', '14.0', '17.0', '16.0', '13.0']
v (float64, 3991 distinct): ['0.0', '8.0', '19.65', '11.61', '27.0', '15.51', '31.7', '51.89', '48.43', '34.87']
l (float64, 55 distinct): ['0.0', '0.03', '0.04', '0.05', '0.06', '0.07', '0.02', '0.08', '0.09', '0.11']
d (float64, 2695 distinct): ['0.0', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0', '9.0', '4.5', '6.0']
i (float64, 4268 distinct): ['0.0', '5.33', '7.74', '9.83', '13.5', '7.75', '12.68', '17.79', '10.8', '13.45']
e (float64, 6978 distinct): ['0.0', '12.0', '17.41', '39.3', '54.0', '31.02', '79.25', '151.35', '49.13', '174.36']
b (float64, 310 distinct): ['0.0', '0.01', '0.02', '0.03', '0.04', '0.06', '0.05', '0.07', '0.08', '0.09']
t (float64, 6761 distinct): ['0.0', '0.67', '0.97', '2.18', '3.0', '1.72', '4.4', '8.41', '9.69', '2.73']
lOCode (float64, 291 distinct): ['0.0', '2.0', '3.0', '4.0', '10.0', '6.0', '9.0', '5.0', '8.0', '13.0']
lOComment (float64, 88 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
lOBlank (float64, 95 distinct): ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
locCodeAndComment (uint8, 30 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '10']
uniq_Op (float64, 68 distinct): ['0.0', '11.0', '12.0', '9.0', '10.0', '4.0', '13.0', '14.0', '7.0', '6.0']
uniq_Opnd (float64, 171 distinct): ['0.0', '6.0', '5.0', '7.0', '4.0', '8.0', '11.0', '10.0', '3.0', '12.0']
total_Op (float64, 581 distinct): ['0.0', '3.0', '5.0', '8.0', '6.0', '9.0', '7.0', '4.0', '10.0', '11.0']
total_Opnd (float64, 468 distinct): ['0.0', '4.0', '2.0', '7.0', '3.0', '6.0', '1.0', '8.0', '5.0', '10.0']
branchCount (float64, 146 distinct): ['1.0', '3.0', '5.0', '7.0', '9.0', '11.0', '13.0', '15.0', '17.0', '19.0']
'''

CONTEXT = "Source Code Quality Prediction: JM1"
TARGET = CuratedTarget(raw_name="defects", task_type=SupervisedTask.BINARY)
