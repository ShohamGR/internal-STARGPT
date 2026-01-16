from tabstar_paper.datasets.curation_objects import CuratedTarget
from tabstar_paper.datasets.objects import SupervisedTask

'''
Dataset Name: taiwanese_bankruptcy_prediction
====
Examples: 6819
====
URL: https://www.openml.org/search?type=data&id=46962
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
- **Original Data Source:** https://doi.org/10.24432/C5004D
- **Reference (please cite)**: Liang, Deron, et al. 'Financial ratios and corporate governance indicators in bankruptcy prediction: A comprehensive study.' European journal of operational research 252.2 (2016): 561-572. https://doi.org/10.1016/j.ejor.2016.01.012
- **Dataset Year:** 2009
- **Dataset Description:** see the reference and the original data source for details.

#### Curation comments by the TabArena team (for code see the [page of the study](https://tabarena.ai/data-tabular-ml-iid-study)):
- We rename the features to be shorter and have no special characters.
- We drop the constant column "Net_Income_Flag".
- Anomaly: the paper introducing the dataset has more features, but these do not seem to be in the dataset. 
- Anomaly: the data is temporal but all features are time-invariant. This might still have a latent, non-verifiable temporal component, which we ignore for our predictive IID task.
====
Target Variable: Bankrupt (category, 2 distinct): ['No', 'Yes']
====
Features:

ROA_C_Before_Interest_Depreciation (float64, 3333 distinct): ['0.4901', '0.5165', '0.5138', '0.4992', '0.5019', '0.4981', '0.4841', '0.501', '0.4804', '0.4951']
ROA_A_Before_Interest_After_Tax (float64, 3151 distinct): ['0.5683', '0.5597', '0.5589', '0.5579', '0.5542', '0.566', '0.5631', '0.5806', '0.5613', '0.5628']
ROA_B_Before_Interest_Depreciation_After_Tax (float64, 3160 distinct): ['0.5525', '0.5388', '0.5582', '0.5515', '0.5523', '0.5436', '0.5434', '0.5507', '0.5405', '0.554']
Operating_Gross_Margin (float64, 3781 distinct): ['0.6065', '0.6058', '0.602', '0.599', '0.6142', '0.6013', '0.6036', '0.6047', '0.6044', '0.6083']
Realized_Sales_Gross_Margin (float64, 3788 distinct): ['0.6047', '0.6058', '0.6026', '0.6003', '0.607', '0.599', '0.6065', '0.6027', '0.6009', '0.5989']
Operating_Profit_Rate (float64, 3376 distinct): ['0.999', '0.999', '0.999', '0.999', '0.999', '0.999', '0.999', '0.999', '0.999', '0.999']
PreTax_Net_Interest_Rate (float64, 3789 distinct): ['0.7974', '0.7974', '0.7974', '0.7974', '0.7974', '0.7974', '0.7975', '0.7975', '0.7974', '0.7975']
AfterTax_Net_Interest_Rate (float64, 3604 distinct): ['0.8093', '0.8094', '0.8093', '0.8093', '0.8093', '0.8093', '0.8093', '0.8093', '0.8093', '0.8093']
NonIndustry_Income_Expenditure_Revenue (float64, 2551 distinct): ['0.3035', '0.3035', '0.3035', '0.3035', '0.3035', '0.3035', '0.3035', '0.3035', '0.3035', '0.3035']
Continuous_Interest_Rate_After_Tax (float64, 3617 distinct): ['0.7817', '0.7816', '0.7816', '0.7816', '0.7816', '0.7816', '0.7816', '0.7816', '0.7816', '0.7816']
Operating_Expense_Rate (float64, 2966 distinct): ['0.0002', '9860000000.0', '5530000000.0', '0.0002', '0.0001', '0.0001', '8950000000.0', '5450000000.0', '8400000000.0', '8480000000.0']
R&D_Expense_Rate (float64, 1536 distinct): ['0.0', '441000000.0', '645000000.0', '815000000.0', '458000000.0', '323000000.0', '968000000.0', '1140000000.0', '1410000000.0', '119000000.0']
Cash_Flow_Rate (float64, 5557 distinct): ['0.4606', '0.461', '0.4644', '0.4625', '0.4639', '0.4603', '0.4636', '0.4644', '0.4623', '0.4632']
InterestBearing_Debt_Interest_Rate (float64, 1080 distinct): ['0.0', '0.0003', '0.0003', '0.0002', '0.0003', '0.0003', '0.0003', '0.0003', '0.0002', '0.0003']
Tax_Rate_A (float64, 2488 distinct): ['0.0', '0.0004', '0.2531', '0.2527', '0.1138', '0.168', '0.0638', '0.2378', '0.2261', '0.0946']
Net_Value_Per_Share_B (float64, 2278 distinct): ['0.177', '0.1746', '0.1846', '0.1755', '0.1752', '0.1829', '0.1811', '0.1806', '0.1753', '0.1771']
Net_Value_Per_Share_A (float64, 2285 distinct): ['0.177', '0.1755', '0.1746', '0.1846', '0.1771', '0.178', '0.1829', '0.1806', '0.1732', '0.1811']
Net_Value_Per_Share_C (float64, 2284 distinct): ['0.177', '0.1746', '0.1755', '0.1846', '0.1732', '0.178', '0.1855', '0.1811', '0.1806', '0.1788']
Persistent_EPS_Last_4_Seasons (float64, 1358 distinct): ['0.2149', '0.2182', '0.2186', '0.2187', '0.2146', '0.219', '0.2185', '0.2201', '0.2192', '0.2251']
Cash_Flow_Per_Share (float64, 1545 distinct): ['0.3192', '0.3226', '0.3205', '0.3213', '0.3178', '0.3215', '0.3198', '0.3193', '0.3211', '0.3243']
Revenue_Per_Share (float64, 3807 distinct): ['0.0178', '0.017', '0.0116', '0.0097', '0.0232', '0.0205', '0.0359', '0.0086', '0.0113', '0.0155']
Operating_Profit_Per_Share (float64, 1236 distinct): ['0.0971', '0.0993', '0.0977', '0.0995', '0.0976', '0.0994', '0.097', '0.1027', '0.1008', '0.0984']
Net_Profit_Before_Tax_Per_Share (float64, 1522 distinct): ['0.1701', '0.17', '0.1742', '0.1709', '0.1747', '0.1737', '0.1717', '0.1759', '0.1703', '0.1704']
Realized_Sales_Gross_Profit_Growth_Rate (float64, 5583 distinct): ['0.0221', '0.0221', '0.0221', '0.0221', '0.0221', '0.0221', '0.0221', '0.0221', '0.022', '0.0221']
Operating_Profit_Growth_Rate (float64, 6249 distinct): ['0.848', '0.848', '0.848', '0.848', '0.8481', '0.848', '0.8481', '0.848', '0.848', '0.848']
AfterTax_Net_Profit_Growth_Rate (float64, 6246 distinct): ['0.6895', '0.6894', '0.6897', '0.6893', '0.6897', '0.6893', '0.6894', '0.6894', '0.6893', '0.6893']
Regular_Net_Profit_Growth_Rate (float64, 6253 distinct): ['0.6894', '0.6894', '0.6895', '0.6894', '0.6896', '0.6896', '0.6894', '0.6893', '0.6898', '0.6894']
Continuous_Net_Profit_Growth_Rate (float64, 6270 distinct): ['0.2176', '0.2176', '0.2176', '0.2176', '0.2176', '0.2176', '0.2176', '0.2176', '0.2176', '0.2176']
Total_Asset_Growth_Rate (float64, 1751 distinct): ['6400000000.0', '6370000000.0', '6470000000.0', '6430000000.0', '6300000000.0', '6440000000.0', '6890000000.0', '6520000000.0', '6610000000.0', '6900000000.0']
Net_Value_Growth_Rate (float64, 4502 distinct): ['0.0004', '0.0004', '0.0004', '0.0005', '0.0005', '0.0005', '0.0005', '0.0005', '0.0005', '0.0005']
Total_Asset_Return_Growth_Rate (float64, 2903 distinct): ['0.2641', '0.2641', '0.264', '0.2639', '0.2641', '0.264', '0.2639', '0.2641', '0.2639', '0.264']
Cash_Reinvestment_Percent (float64, 3599 distinct): ['0.3759', '0.3754', '0.3866', '0.3765', '0.3765', '0.3849', '0.3789', '0.3794', '0.3803', '0.3823']
Current_Ratio (float64, 6132 distinct): ['0.0121', '0.0059', '0.0132', '0.0092', '0.0071', '0.0069', '0.0071', '0.0061', '0.0122', '0.0097']
Quick_Ratio (float64, 6094 distinct): ['0.0054', '0.0072', '0.0069', '0.0096', '0.0021', '0.0109', '0.0061', '0.0076', '0.0077', '0.0062']
Interest_Expense_Ratio (float64, 3794 distinct): ['0.6306', '0.6306', '0.6306', '0.6306', '0.6306', '0.6306', '0.6306', '0.6306', '0.6306', '0.6306']
Total_Debt_to_Net_Worth (float64, 5518 distinct): ['0.0015', '0.0032', '0.0034', '0.0014', '0.0052', '0.001', '0.007', '0.004', '0.0092', '0.0011']
Debt_Ratio_Percent (float64, 4208 distinct): ['0.0892', '0.1282', '0.1129', '0.1402', '0.1195', '0.1155', '0.1234', '0.1067', '0.0784', '0.1069']
Net_Worth_to_Assets (float64, 4208 distinct): ['0.9108', '0.8718', '0.8871', '0.8598', '0.8805', '0.8845', '0.8766', '0.8933', '0.9216', '0.8931']
LongTerm_Fund_Suitability_Ratio_A (float64, 6523 distinct): ['0.0047', '0.0051', '0.0054', '0.0053', '0.0051', '0.0055', '0.0055', '0.0051', '0.0051', '0.0054']
Borrowing_Dependency (float64, 4338 distinct): ['0.3696', '0.3696', '0.3696', '0.3697', '0.3697', '0.3706', '0.3707', '0.3698', '0.3697', '0.3716']
Contingent_Liabilities_to_Net_Worth (float64, 1855 distinct): ['0.0054', '0.0055', '0.0056', '0.0057', '0.0054', '0.0055', '0.0058', '0.0062', '0.0054', '0.0058']
Operating_Profit_to_PaidIn_Capital (float64, 4423 distinct): ['0.0979', '0.0993', '0.1032', '0.1097', '0.0994', '0.0984', '0.1132', '0.1048', '0.0995', '0.0951']
Net_Profit_Before_Tax_to_PaidIn_Capital (float64, 4785 distinct): ['0.1784', '0.1706', '0.169', '0.1694', '0.1714', '0.1718', '0.1722', '0.179', '0.1801', '0.1794']
Inventory_Accounts_Receivable_to_Net_Value (float64, 5289 distinct): ['0.3937', '0.4002', '0.3963', '0.4003', '0.399', '0.4005', '0.4016', '0.3951', '0.3988', '0.3998']
Total_Asset_Turnover (float64, 381 distinct): ['0.0795', '0.1034', '0.093', '0.1139', '0.1064', '0.072', '0.1259', '0.1004', '0.099', '0.069']
Accounts_Receivable_Turnover (float64, 1593 distinct): ['0.0007', '0.0007', '0.0008', '0.0009', '0.0006', '0.0007', '0.0007', '0.0007', '0.0006', '0.0007']
Average_Collection_Days (float64, 5451 distinct): ['0.0', '0.0073', '0.0076', '0.0078', '0.0057', '0.0093', '0.0062', '0.008', '0.0031', '0.0069']
Inventory_Turnover_Rate (float64, 2397 distinct): ['19100000.0', '812000000.0', '8370000000.0', '8100000000.0', '5690000000.0', '8460000000.0', '8500000000.0', '8940000000.0', '0.0001', '8750000000.0']
Fixed_Assets_Turnover_Frequency (float64, 2451 distinct): ['0.0001', '0.0001', '469000000.0', '8470000000.0', '6280000000.0', '787000000.0', '9380000000.0', '6880000000.0', '0.0001', '9150000000.0']
Net_Worth_Turnover_Rate (float64, 741 distinct): ['0.0284', '0.0261', '0.0216', '0.0265', '0.0224', '0.0245', '0.0185', '0.0252', '0.0281', '0.0226']
Revenue_Per_Person (float64, 5667 distinct): ['0.0136', '0.0238', '0.0076', '0.0082', '0.0087', '0.0254', '0.0128', '0.0092', '0.0065', '0.0113']
Operating_Profit_Per_Person (float64, 3023 distinct): ['0.3945', '0.3923', '0.3952', '0.3932', '0.3949', '0.394', '0.3938', '0.3945', '0.3942', '0.3954']
Allocation_Rate_Per_Person (float64, 6768 distinct): ['0.0', '0.0056', '0.0002', '0.0073', '0.0167', '0.0004', '0.0057', '0.0053', '0.0064', '0.0145']
Working_Capital_to_Total_Assets (float64, 6819 distinct): ['0.8141', '0.7966', '0.7985', '0.7897', '0.8665', '0.7647', '0.728', '0.9011', '0.7667', '0.7242']
Quick_Assets_to_Total_Assets (float64, 6819 distinct): ['0.3792', '0.5432', '0.3721', '0.398', '0.3934', '0.1987', '0.2295', '0.6094', '0.3111', '0.3467']
Current_Assets_to_Total_Assets (float64, 6819 distinct): ['0.3891', '0.67', '0.4155', '0.4063', '0.4722', '0.2693', '0.2739', '0.642', '0.3946', '0.393']
Cash_to_Total_Assets (float64, 6819 distinct): ['0.2145', '0.0788', '0.039', '0.0065', '0.0614', '0.1038', '0.0938', '0.1767', '0.1368', '0.0478']
Quick_Assets_to_Current_Liability (float64, 6819 distinct): ['0.0139', '0.0063', '0.009', '0.0088', '0.0429', '0.0053', '0.0037', '0.0371', '0.0054', '0.0041']
Cash_to_Current_Liability (float64, 6816 distinct): ['8870000000.0', '7510000000.0', '4610000000.0', '0.0227', '0.0055', '0.0027', '0.0004', '0.0191', '0.008', '0.0045']
Current_Liability_to_Assets (float64, 6819 distinct): ['0.0498', '0.1557', '0.0754', '0.0825', '0.0166', '0.0684', '0.111', '0.0299', '0.1046', '0.1518']
Operating_Funds_to_Liability (float64, 6819 distinct): ['0.389', '0.3366', '0.3447', '0.3835', '0.3852', '0.3529', '0.3477', '0.4009', '0.3085', '0.3402']
Inventory_to_Working_Capital (float64, 6593 distinct): ['0.277', '0.277', '0.2774', '0.2777', '0.2774', '0.277', '0.2772', '0.2772', '0.2767', '0.277']
Inventory_to_Current_Liability (float64, 6590 distinct): ['0.0', '8790000000.0', '5070000000.0', '5280000000.0', '0.0025', '0.0021', '0.0068', '0.0147', '0.0143', '0.0134']
Current_Liabilities_to_Liability (float64, 6627 distinct): ['1.0', '0.9808', '0.4382', '0.9915', '0.6371', '0.9777', '0.6122', '0.3828', '0.7315', '0.9238']
Working_Capital_to_Equity (float64, 6819 distinct): ['0.735', '0.7359', '0.7351', '0.7341', '0.7371', '0.7333', '0.7301', '0.7388', '0.7338', '0.7293']
Current_Liabilities_to_Equity (float64, 6819 distinct): ['0.3277', '0.3349', '0.3294', '0.3291', '0.3267', '0.3303', '0.3319', '0.327', '0.3341', '0.3356']
LongTerm_Liability_to_Current_Assets (float64, 4249 distinct): ['0.0', '579000000.0', '279000000.0', '0.0312', '0.0359', '0.011', '0.0001', '0.1593', '0.0174', '0.0105']
Retained_Earnings_to_Total_Assets (float64, 6819 distinct): ['0.9538', '0.9298', '0.9378', '0.9462', '0.9463', '0.9312', '0.9223', '0.9507', '0.8661', '0.9201']
Total_Income_to_Total_Expense (float64, 6819 distinct): ['0.0026', '0.0022', '0.0025', '0.0025', '0.0027', '0.0022', '0.0018', '0.0024', '0.0021', '0.0019']
Total_Expense_to_Assets (float64, 6819 distinct): ['0.031', '0.0192', '0.013', '0.0241', '0.0094', '0.011', '0.0106', '0.0337', '0.1163', '0.0204']
Current_Asset_Turnover_Rate (float64, 6261 distinct): ['8580000000.0', '9980000000.0', '9470000000.0', '9950000000.0', '9660000000.0', '8470000000.0', '9720000000.0', '7320000000.0', '7410000000.0', '7450000000.0']
Quick_Asset_Turnover_Rate (float64, 5377 distinct): ['6460000000.0', '8590000000.0', '9160000000.0', '9590000000.0', '5830000000.0', '6440000000.0', '8480000000.0', '9910000000.0', '9260000000.0', '9840000000.0']
Working_Capital_Turnover_Rate (float64, 6819 distinct): ['0.594', '0.5939', '0.594', '0.5939', '0.5941', '0.5939', '0.5939', '0.5941', '0.5939', '0.5939']
Cash_Turnover_Rate (float64, 4023 distinct): ['1940000000.0', '3020000000.0', '1930000000.0', '2060000000.0', '1750000000.0', '2870000000.0', '5450000000.0', '4550000000.0', '2110000000.0', '2220000000.0']
Cash_Flow_to_Sales (float64, 6819 distinct): ['0.6715', '0.6716', '0.6716', '0.6716', '0.6716', '0.6715', '0.6716', '0.6715', '0.6715', '0.6716']
Fixed_Assets_to_Assets (float64, 6814 distinct): ['0.0', '0.1262', '0.4895', '0.6192', '0.0612', '0.0891', '0.1393', '0.4669', '0.1159', '0.721']
Current_Liability_to_Liability (float64, 6627 distinct): ['1.0', '0.9808', '0.4382', '0.9915', '0.6371', '0.9777', '0.6122', '0.3828', '0.7315', '0.9238']
Current_Liability_to_Equity (float64, 6819 distinct): ['0.3277', '0.3349', '0.3294', '0.3291', '0.3267', '0.3303', '0.3319', '0.327', '0.3341', '0.3356']
Equity_to_LongTerm_Liability (float64, 4251 distinct): ['0.1109', '0.113', '0.1141', '0.1154', '0.111', '0.1367', '0.116', '0.1273', '0.1172', '0.128']
Cash_Flow_to_Total_Assets (float64, 6819 distinct): ['0.5966', '0.6587', '0.6216', '0.6413', '0.662', '0.6004', '0.6338', '0.6126', '0.628', '0.6453']
Cash_Flow_to_Liability (float64, 6819 distinct): ['0.435', '0.4618', '0.4544', '0.4588', '0.4778', '0.4528', '0.4577', '0.4345', '0.4573', '0.4596']
CFO_to_Assets (float64, 6819 distinct): ['0.6407', '0.544', '0.5793', '0.6763', '0.6024', '0.6308', '0.5975', '0.6252', '0.3646', '0.5623']
Cash_Flow_to_Equity (float64, 6819 distinct): ['0.3094', '0.3177', '0.3114', '0.3144', '0.3165', '0.3055', '0.3129', '0.3114', '0.3106', '0.3151']
Current_Liability_to_Current_Assets (float64, 6819 distinct): ['0.0198', '0.0363', '0.0282', '0.0315', '0.0054', '0.0392', '0.0626', '0.0071', '0.0412', '0.0601']
Liability_Assets_Flag (uint8, 2 distinct): ['0', '1']
Net_Income_to_Total_Assets (float64, 6819 distinct): ['0.8544', '0.7913', '0.8154', '0.8405', '0.8219', '0.7932', '0.7687', '0.8254', '0.6946', '0.7534']
Total_Assets_to_GNP_Price (float64, 6818 distinct): ['0.0037', '0.0078', '0.0047', '0.007', '0.0021', '0.0015', '0.0041', '0.0055', '0.0042', '0.1132']
NoCredit_Interval (float64, 6819 distinct): ['0.624', '0.6238', '0.6241', '0.624', '0.6277', '0.6238', '0.624', '0.6245', '0.6237', '0.625']
Gross_Profit_to_Sales (float64, 6816 distinct): ['0.6651', '0.6262', '0.6035', '0.6027', '0.6161', '0.6097', '0.5965', '0.5884', '0.6252', '0.6475']
Net_Income_to_Stockholders_Equity (float64, 6819 distinct): ['0.8433', '0.8396', '0.8415', '0.8429', '0.8413', '0.8397', '0.8376', '0.8416', '0.8266', '0.8354']
Liability_to_Equity (float64, 6819 distinct): ['0.276', '0.2822', '0.2788', '0.2772', '0.2753', '0.2833', '0.281', '0.2755', '0.2872', '0.2838']
DFL (float64, 6240 distinct): ['0.0268', '0.0268', '0.0268', '0.0268', '0.0284', '0.0267', '0.0267', '0.0268', '0.0273', '0.0269']
Interest_Coverage_Ratio (float64, 6240 distinct): ['0.5652', '0.5652', '0.5653', '0.565', '0.5682', '0.5647', '0.5646', '0.5652', '0.5667', '0.5655']
Equity_to_Liability (float64, 6819 distinct): ['0.084', '0.0231', '0.034', '0.049', '0.1594', '0.0214', '0.0255', '0.1336', '0.018', '0.0208']
'''

CONTEXT = "Taiwanese companies bankruptcy prediction"
TARGET = CuratedTarget(raw_name="Bankrupt", task_type=SupervisedTask.BINARY)
