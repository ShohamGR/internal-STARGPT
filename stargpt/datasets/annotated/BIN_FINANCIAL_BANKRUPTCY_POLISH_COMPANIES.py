from tabstar_paper.datasets.curation_objects import CuratedTarget
from tabstar_paper.datasets.objects import SupervisedTask

'''
Dataset Name: polish_companies_bankruptcy
====
Examples: 5910
====
URL: https://www.openml.org/search?type=data&id=46950
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
- **Original Data Source:** https://doi.org/10.24432/C5F600
- **Reference (please cite)**: Zieba, Maciej, Sebastian K. Tomczak, and Jakub M. Tomczak. 'Ensemble boosted trees with synthetic features generation in application to bankruptcy prediction.' Expert systems with applications 58 (2016): 93-101. https://doi.org/10.1016/j.eswa.2016.04.001
- **Dataset Year:** 2010
- **Dataset Description:** see the reference and the original data source for details.

#### Curation comments by the TabArena team (for code see the [page of the study](https://tabarena.ai/data-tabular-ml-iid-study)):
- We only use data from year 5, because it is the newest data and the target is bankruptcy status after only 1 year.
- We created semantically meaningful feature names.
- Anomaly: the data contains a lot of features created by feature engineering.
====
Target Variable: company_bankrupt (category, 2 distinct): ['No', 'Yes']
====
Features:

net_profit_to_total_assets (float64, 5622 distinct): ['0.0', '0.1535', '0.0003', '0.1271', '0.0462', '0.2229', '0.1472', '0.0061', '0.0495', '-0.0365']
total_liabilities_to_total_assets (float64, 5619 distinct): ['0.0', '0.6077', '0.66', '0.2833', '0.7195', '0.2506', '0.402', '0.1307', '0.3853', '0.2228']
working_capital_to_total_assets (float64, 5653 distinct): ['1.0', '0.0752', '0.5175', '0.5998', '0.1249', '0.2003', '0.1567', '0.3864', '0.656', '0.5067']
current_assets_to_short_term_liabilities (float64, 5466 distinct): ['1.5094', '1.1072', '1.029', '1.9393', '1.1534', '1.287', '1.0459', '1.2003', '1.4348', '1.4858']
liquidity_days_ratio (float64, 5778 distinct): ['0.0', '33.352', '-121.81', '-65.399', '-136.32', '-20.214', '-40.108', '117.75', '3.9122', '331.46']
retained_earnings_to_total_assets (float64, 3539 distinct): ['0.0', '0.1352', '-0.1508', '0.3972', '0.0003', '0.0924', '0.1225', '0.0924', '0.1726', '-0.0198']
ebit_to_total_assets (float64, 5652 distinct): ['0.0', '0.2044', '0.1591', '-0.0365', '0.2314', '0.0324', '0.0617', '0.1468', '0.1565', '-0.1384']
book_value_equity_to_total_liabilities (float64, 5635 distinct): ['1.1634', '1.2376', '1.0819', '1.6501', '2.7603', '0.7327', '0.3363', '2.1171', '1.0968', '1.9198']
sales_to_total_assets (float64, 4823 distinct): ['1.0277', '1.0488', '1.0117', '1.0392', '1.0506', '1.0376', '1.081', '1.0219', '1.0482', '1.0077']
equity_to_total_assets (float64, 5612 distinct): ['1.0', '0.828', '0.7167', '0.7494', '0.5704', '0.2406', '0.7477', '0.5374', '0.4229', '0.3497']
extended_profit_to_total_assets (float64, 5667 distinct): ['0.0', '0.1482', '0.0931', '0.2163', '0.1426', '0.1647', '0.2571', '0.115', '0.1341', '0.1668']
gross_profit_to_short_term_liabilities (float64, 5685 distinct): ['0.465', '-0.2794', '0.3326', '0.7178', '0.1001', '-0.1299', '0.3986', '0.2061', '0.4745', '0.1086']
gross_profit_plus_depreciation_to_sales (float64, 5659 distinct): ['0.0', '0.1226', '0.1285', '0.0263', '0.106', '0.1097', '0.1131', '0.0855', '0.169', '0.162']
gross_profit_plus_interest_to_total_assets (float64, 5652 distinct): ['0.0', '0.2044', '0.1591', '-0.0365', '0.2314', '0.0324', '0.0617', '0.1468', '0.1565', '-0.1384']
liabilities_days_ratio (float64, 5690 distinct): ['0.0', '305.44', '448.06', '765.4', '361.35', '589.77', '380.1', '2069.4', '1696.8', '1302.2']
gross_profit_plus_depreciation_to_total_liabilities (float64, 5699 distinct): ['0.8146', '1.0101', '0.1764', '1.195', '0.6189', '0.4769', '0.2604', '1.1534', '0.3436', '0.1949']
total_assets_to_total_liabilities (float64, 5448 distinct): ['1.9046', '1.3363', '1.5152', '1.5508', '1.8418', '2.6501', '4.4883', '1.2117', '1.309', '1.1471']
gross_profit_to_total_assets (float64, 5652 distinct): ['0.0', '0.2044', '0.1591', '-0.0365', '0.2314', '0.0324', '0.0617', '0.1468', '0.1565', '-0.1384']
gross_profit_to_sales (float64, 5675 distinct): ['0.0', '0.0898', '0.0991', '-0.1492', '0.0663', '0.0378', '0.1081', '0.1142', '0.1394', '0.0222']
inventory_days_ratio (float64, 5392 distinct): ['0.0', '8.9365', '82.843', '23.863', '1.5522', '21.568', '16.639', '114.58', '69.508', '32.642']
sales_growth_ratio (float64, 4420 distinct): ['1.121', '1.2193', '1.0058', '1.0022', '1.1892', '1.3353', '1.1882', '1.1906', '1.1148', '1.0942']
operating_profit_to_total_assets (float64, 5144 distinct): ['0.0', '0.111', '0.1706', '0.1168', '0.1396', '0.1608', '0.0353', '0.0682', '0.1235', '0.1637']
net_profit_to_sales (float64, 5649 distinct): ['0.0', '0.1914', '0.0002', '0.0701', '0.0532', '0.0792', '0.0008', '0.0552', '-0.1492', '0.0142']
three_year_gross_profit_to_total_assets (float64, 5551 distinct): ['0.0', '-0.2641', '0.3742', '0.4619', '1.4774', '0.1345', '0.1304', '0.1248', '0.735', '0.1928']
equity_minus_share_capital_to_total_assets (float64, 5666 distinct): ['0.7017', '0.0', '0.5013', '0.7256', '0.4726', '0.5262', '0.6417', '0.6626', '0.5942', '0.3994']
net_profit_plus_depreciation_to_total_liabilities (float64, 5667 distinct): ['0.0', '0.1061', '0.671', '0.139', '1.0898', '0.4282', '0.1932', '0.1939', '0.5512', '0.6572']
operating_profit_to_financial_expenses (float64, 4985 distinct): ['0.0', '1.3454', '1.397', '2.8096', '2.6295', '7.5518', '3.0986', '2.4728', '0.346', '7.1814']
working_capital_to_fixed_assets (float64, 5611 distinct): ['1.2599', '0.1804', '1.0404', '0.0977', '3.3585', '-0.3352', '10.419', '1.2433', '0.6721', '-0.195']
log_total_assets (float64, 5214 distinct): ['4.1709', '3.4531', '4.8609', '4.1917', '4.402', '4.6746', '4.1152', '3.8214', '3.6421', '3.5293']
net_liabilities_to_sales (float64, 5622 distinct): ['-0.0421', '0.1442', '0.4701', '0.6622', '0.6125', '0.2685', '0.2302', '0.1227', '0.5552', '0.0']
gross_profit_plus_interest_to_sales (float64, 5699 distinct): ['0.0', '0.1128', '0.0663', '0.0053', '0.0738', '0.0173', '0.0992', '0.0371', '0.1142', '0.108']
current_liabilities_days_ratio (float64, 5510 distinct): ['0.0', '90.34', '109.97', '100.89', '76.202', '57.341', '140.12', '124.9', '48.78', '135.24']
operating_expenses_to_short_term_liabilities (float64, 5592 distinct): ['0.0', '4.0403', '5.1719', '3.4352', '3.8291', '3.0054', '3.264', '5.1785', '4.2849', '2.0725']
operating_expenses_to_total_liabilities (float64, 5688 distinct): ['0.0', '0.2152', '3.5357', '12.55', '0.1439', '3.5638', '2.1971', '1.4689', '5.0243', '2.1311']
sales_profit_to_total_assets (float64, 5652 distinct): ['0.0', '0.0', '0.0126', '0.0', '0.1463', '0.157', '0.1018', '0.101', '0.0569', '0.0926']
total_sales_to_total_assets (float64, 5420 distinct): ['2.0409', '1.2038', '1.3215', '1.1946', '1.5765', '1.5483', '1.3512', '1.3137', '1.7395', '1.4214']
current_assets_minus_inventories_to_long_term_liabilities (float64, 3260 distinct): ['2.4927', '4.1023', '1.5338', '2.7727', '225.42', '18.721', '2.1297', '254.14', '11.675', '8.8726']
constant_capital_to_total_assets (float64, 5571 distinct): ['1.0', '0.828', '0.8416', '0.7772', '0.832', '0.759', '0.6142', '0.443', '0.8452', '0.8079']
sales_profit_to_sales (float64, 5655 distinct): ['0.0', '0.1036', '0.0', '0.0978', '0.002', '0.0', '0.1247', '0.0144', '0.0206', '0.0002']
liquid_assets_to_short_term_liabilities (float64, 5699 distinct): ['0.6165', '0.0', '0.3246', '1.3923', '0.1471', '0.2736', '1.2077', '0.0867', '0.4643', '22.065']
liabilities_to_adjusted_operating_profit (float64, 5593 distinct): ['0.0', '0.2701', '0.2925', '0.1466', '0.0381', '0.1382', '0.1093', '0.1128', '0.2253', '0.0405']
operating_profit_to_sales (float64, 5158 distinct): ['0.0', '0.0', '0.1002', '0.1052', '0.0181', '-0.1247', '0.0288', '0.0733', '0.0095', '0.002']
receivables_plus_inventory_turnover_days (float64, 5347 distinct): ['0.0', '117.68', '119.59', '105.27', '111.2', '134.82', '150.06', '117.8', '121.24', '143.53']
receivables_days_ratio (float64, 5614 distinct): ['0.0', '119.61', '109.6', '41.469', '54.412', '21.74', '108.58', '50.144', '103.3', '41.008']
net_profit_to_inventory (float64, 5420 distinct): ['0.0', '0.5952', '1.5193', '-1.0171', '0.161', '-35.075', '0.2693', '0.349', '0.132', '0.1488']
current_assets_minus_inventory_to_short_term_liabilities (float64, 5550 distinct): ['1.3217', '2.0621', '1.0043', '1.1234', '2.0599', '1.6191', '1.1262', '1.2039', '1.0492', '1.6204']
inventory_days_cost_ratio (float64, 5394 distinct): ['0.0', '126.64', '39.992', '24.832', '33.904', '140.3', '52.379', '27.156', '45.975', '110.29']
ebitda_to_total_assets (float64, 5613 distinct): ['0.0', '-0.0038', '0.1267', '0.0104', '0.1384', '0.1017', '-0.0179', '0.026', '0.0084', '-0.0381']
ebitda_to_sales (float64, 5601 distinct): ['0.0', '-0.0089', '0.1061', '0.0037', '0.0862', '-0.0', '0.0111', '-0.0318', '0.1194', '-0.0198']
current_assets_to_total_liabilities (float64, 5506 distinct): ['1.0454', '1.0375', '1.2891', '0.551', '1.0433', '1.1042', '1.2161', '1.2581', '2.9458', '1.5094']
short_term_liabilities_to_total_assets (float64, 5576 distinct): ['0.0', '0.1548', '0.1307', '0.1642', '0.3575', '0.6488', '0.2391', '0.3019', '0.7195', '0.1805']
short_term_liabilities_days_cost_ratio (float64, 5499 distinct): ['0.0', '0.2475', '0.1571', '0.15', '0.2334', '0.1949', '0.1097', '0.1082', '0.3674', '0.1933']
equity_to_fixed_assets (float64, 5462 distinct): ['2.2499', '2.3016', '0.9733', '1.0655', '1.2624', '1.057', '1.4562', '1.0057', '1.0391', '1.5962']
constant_capital_to_fixed_assets (float64, 5349 distinct): ['2.2499', '1.0371', '1.1296', '1.1804', '1.6298', '1.3109', '1.0927', '1.2642', '1.39', '1.0688']
working_capital_absolute (float64, 5718 distinct): ['13593.0', '1.4858', '4256.3', '2443.9', '10361.0', '-11216.0', '1797.0', '183340.0', '10445.0', '2019.8']
gross_margin (float64, 5646 distinct): ['1.0', '0.124', '0.2027', '0.1036', '0.1478', '0.1037', '0.1138', '0.1695', '0.0978', '0.184']
adjusted_liquidity_ratio (float64, 5572 distinct): ['0.0', '0.1581', '0.0062', '0.1864', '0.1209', '0.3068', '0.1122', '0.1034', '0.1636', '0.1594']
total_costs_to_total_sales (float64, 5115 distinct): ['1.0053', '0.9943', '1.0115', '1.0179', '1.0505', '1.0173', '1.0438', '1.0', '0.9641', '1.0107']
long_term_liabilities_to_equity (float64, 3254 distinct): ['0.0', '0.1278', '0.1007', '0.0026', '0.4708', '0.7828', '0.6886', '0.1929', '0.1432', '0.0446']
inventory_turnover_ratio (float64, 5369 distinct): ['16.923', '15.296', '40.844', '3.1856', '17.114', '235.16', '4.4059', '3.8537', '5.2512', '11.257']
receivables_turnover_ratio (float64, 5611 distinct): ['10.096', '8.9007', '9.4409', '10.809', '4.3754', '5.695', '5.9772', '7.2732', '6.7081', '16.789']
short_term_liabilities_days_ratio (float64, 5559 distinct): ['0.0', '102.89', '104.08', '50.409', '116.45', '117.87', '141.76', '107.84', '108.84', '38.428']
sales_to_short_term_liabilities (float64, 5577 distinct): ['7.2408', '14.691', '5.6358', '2.3307', '3.1343', '2.5748', '3.6796', '1.9972', '3.3535', '2.1269']
sales_to_fixed_assets (float64, 5548 distinct): ['8.9884', '1.2082', '1.8704', '2.682', '16.589', '3.358', '1.6959', '2.5157', '6.7586', '9.233']
'''

CONTEXT = "Polish Companies Bankruptcy"
TARGET = CuratedTarget(raw_name="company_bankrupt", task_type=SupervisedTask.BINARY)
