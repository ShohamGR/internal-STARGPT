from tabstar_paper.datasets.curation_objects import CuratedTarget
from tabstar_paper.datasets.objects import SupervisedTask

'''
Dataset Name: airfoil_self_noise
====
Examples: 1503
====
URL: https://www.openml.org/search?type=data&id=46904
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
- **Original Data Source:** https://doi.org/10.24432/C5VW2C
- **Reference (please cite)**: Brooks, Thomas F., D. Stuart Pope, and Michael A. Marcolini. Airfoil self-noise and prediction. No. L-16528. 1989. https://ntrs.nasa.gov/citations/19890016302
- **Dataset Year:** 2014
- **Dataset Description:** see the reference and the original data source for details.

#### Curation comments by the TabArena team (for code see the [page of the study](https://tabarena.ai/data-tabular-ml-iid-study)):
- N/A

====
URL: https://www.openml.org/search?type=data&id=44957
====
Description: **Data Description**

NASA data set, obtained from a series of aerodynamic and acoustic tests of two and three-dimensional airfoil blade sections conducted in an anechoic wind tunnel.

It comprises different size NACA 0012 airfoils at various wind tunnel speeds and angles of attack. The span of the airfoil and the observer position were the same in all of the experiments.

The task is to predict the (scaled) self noise.

**Attribute Description**

1. *frequency* - in Hertzs
2. *angle_of_attack* - in degrees
3. *chord_length* - in meters
4. *free_stream_velocity* - in meters per second
5. *displacement_thickness* - in meters
6. *sound_pressure* - in decibels (target feature)


====
Target Variable: sound_pressure (numeric, 1456 distinct): ['127.315', '126.54', '129.395', '126.805', '123.742', '119.737', '120.189', '131.955', '130.307', '118.134']
====
Features:

frequency (numeric, 21 distinct): ['2000.0', '2500.0', '1600.0', '3150.0', '4000.0', '1250.0', '1000.0', '800.0', '5000.0', '6300.0']
angle_of_attack (numeric, 27 distinct): ['0.0', '4.0', '15.4', '12.3', '7.3', '9.9', '17.4', '3.0', '2.0', '9.5']
chord_length (numeric, 6 distinct): ['0.0254', '0.1524', '0.2286', '0.1016', '0.0508', '0.3048']
free_stream_velocity (numeric, 4 distinct): ['39.6', '71.3', '31.7', '55.5']
displacement_thickness (numeric, 105 distinct): ['0.0053', '0.0031', '0.0033', '0.005', '0.0264', '0.0039', '0.0161', '0.013', '0.0122', '0.004']
'''


CONTEXT = "NASA Aerodynamic and Acoustic Tests of Airfoil Blade Sections"
TARGET = CuratedTarget(raw_name="scaled-sound-pressure", task_type=SupervisedTask.REGRESSION)
