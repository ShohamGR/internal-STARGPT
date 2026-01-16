from tabstar_paper.datasets.curation_objects import CuratedTarget
from tabstar_paper.datasets.objects import SupervisedTask

'''
Dataset Name: coil2000_insurance_policies
====
Examples: 9822
====
URL: https://www.openml.org/search?type=data&id=46916
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
- **Original Data Source:** https://doi.org/10.24432/C5630S
- **Reference (please cite)**: Van Der Putten, Peter, and Maarten van Someren. CoIL challenge 2000: The insurance company case. Technical Report 200009, Leiden Institute of Advanced Computer Science, Universiteit van Leiden. Available from:< http://www. liacs. nl/putten/library/cc2000, 2000.
- **Dataset Year:** 2000
- **Dataset Description:** see the reference and the original data source for details.

#### Curation comments by the TabArena team (for code see the [page of the study](https://tabarena.ai/data-tabular-ml-iid-study)):
- We created semantic meaningful names for the features.
- We combined the original training and validation data into one new dataset.
- We reversed the ordinal encoding of the original data where possible.
- Anomaly: the data has 15% duplicates.
====
Target Variable: MobileHomePolicy (category, 2 distinct): ['No', 'Yes']
====
Features:

customerSubtype (category, 39 distinct): ['Lower class large families', 'Traditional families', 'Middle class families', 'Large religous families', 'Modern, complete families', 'High status seniors', 'Young and rising', "Couples with teens 'Married with children'", 'Village families', 'Mixed rurals']
numberOfHouses (uint8, 9 distinct): ['1', '2', '3', '7', '4', '6', '5', '10', '8']
avgSizeHousehold (uint8, 6 distinct): ['3', '2', '4', '1', '5', '6']
avgAge (object, 6 distinct): ['40-50 years', '30-40 years', '50-60 years', '60-70 years', '20-30 years', '70-80 years']
customerMainType (category, 10 distinct): ['Family with grown ups', 'Average Family', 'Conservative families', 'Successful hedonists', 'Living well', 'Retired and Religeous', 'Driven Growers', 'Farmers', 'Cruising Seniors', 'Career Loners']
romanCatholic (category, 10 distinct): ['0%', '1 - 10%', '11 - 23%', '24 - 36%', '37 - 49%', '50 - 62%', '63 - 75%', '76 - 88%', '100%', '89 - 99%']
protestant (uint8, 10 distinct): ['4', '5', '6', '3', '7', '2', '9', '1', '0', '8']
otherReligion (uint8, 6 distinct): ['0', '1', '2', '3', '4', '5']
noReligion (uint8, 10 distinct): ['3', '4', '2', '5', '0', '6', '1', '7', '9', '8']
married (uint8, 10 distinct): ['7', '6', '5', '9', '8', '4', '3', '2', '1', '0']
livingTogether (uint8, 8 distinct): ['0', '1', '2', '3', '4', '5', '6', '7']
otherRelation (uint8, 10 distinct): ['2', '0', '3', '4', '1', '5', '6', '7', '9', '8']
singles (uint8, 10 distinct): ['0', '2', '1', '3', '4', '5', '6', '7', '8', '9']
householdWithoutChildren (uint8, 10 distinct): ['3', '4', '2', '5', '0', '1', '6', '7', '9', '8']
householdWithChildren (uint8, 10 distinct): ['4', '5', '3', '6', '2', '7', '1', '8', '9', '0']
highLevelEducation (uint8, 10 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
mediumLevelEducation (uint8, 10 distinct): ['4', '3', '2', '5', '0', '1', '6', '7', '9', '8']
lowerLevelEducation (uint8, 10 distinct): ['5', '6', '4', '3', '2', '7', '9', '0', '8', '1']
highStatus (uint8, 10 distinct): ['0', '2', '1', '3', '4', '5', '6', '7', '9', '8']
entrepreneur (uint8, 6 distinct): ['0', '1', '2', '5', '3', '4']
farmer (uint8, 10 distinct): ['0', '1', '2', '3', '4', '5', '6', '8', '9', '7']
middleManagement (uint8, 10 distinct): ['2', '3', '4', '0', '5', '1', '6', '7', '9', '8']
skilledLabourers (uint8, 10 distinct): ['2', '0', '3', '1', '4', '5', '6', '7', '8', '9']
unskilledLabourers (uint8, 10 distinct): ['2', '3', '1', '0', '4', '5', '6', '7', '9', '8']
socialClassA (uint8, 10 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '9', '8']
socialClassB1 (uint8, 10 distinct): ['2', '1', '0', '3', '4', '5', '6', '9', '8', '7']
socialClassB2 (uint8, 10 distinct): ['2', '3', '0', '1', '4', '5', '6', '8', '7', '9']
socialClassC (uint8, 10 distinct): ['5', '4', '3', '2', '6', '0', '1', '7', '9', '8']
socialClassD (uint8, 10 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '9', '8']
rentedHouse (uint8, 10 distinct): ['0', '9', '2', '3', '4', '8', '5', '1', '7', '6']
homeOwners (uint8, 10 distinct): ['9', '0', '7', '6', '5', '1', '4', '8', '2', '3']
oneCar (uint8, 10 distinct): ['6', '7', '5', '9', '4', '8', '3', '2', '0', '1']
twoCars (uint8, 9 distinct): ['0', '2', '1', '3', '4', '5', '6', '7', '9']
noCar (uint8, 10 distinct): ['2', '0', '3', '1', '4', '5', '6', '7', '9', '8']
nationalHealthService (uint8, 10 distinct): ['7', '5', '6', '9', '8', '4', '2', '3', '0', '1']
privateHealthInsurance (uint8, 10 distinct): ['2', '4', '0', '3', '1', '5', '7', '6', '9', '8']
incomeLessThan30k (uint8, 10 distinct): ['0', '2', '3', '1', '4', '5', '6', '7', '9', '8']
income30To45k (uint8, 10 distinct): ['4', '3', '5', '2', '0', '6', '1', '7', '9', '8']
income45To75k (uint8, 10 distinct): ['3', '2', '4', '0', '1', '5', '6', '7', '9', '8']
income75To122k (uint8, 10 distinct): ['0', '1', '2', '3', '4', '5', '6', '8', '9', '7']
incomeAbove123k (uint8, 9 distinct): ['0', '1', '2', '3', '4', '5', '6', '9', '7']
averageIncome (uint8, 10 distinct): ['3', '4', '5', '2', '6', '7', '8', '1', '0', '9']
purchasingPowerClass (uint8, 8 distinct): ['3', '6', '4', '5', '1', '7', '2', '8']
contributionPrivateThirdPartyInsurance (object, 4 distinct): ['f 0', 'f 50 - 99', 'f 1 - 49', 'f 100 - 199']
contributionThirdPartyInsuranceFirms (uint8, 7 distinct): ['0', '2', '3', '4', '1', '6', '5']
contributionThirdPartyInsuranceAgriculture (uint8, 5 distinct): ['0', '4', '3', '2', '1']
contributionCarPolicies (uint8, 7 distinct): ['0', '6', '5', '7', '8', '4', '9']
contributionDeliveryVanPolicies (uint8, 4 distinct): ['0', '6', '5', '7']
contributionMotorcycleScooterPolicies (uint8, 6 distinct): ['0', '4', '6', '5', '3', '7']
contributionLorryPolicies (uint8, 5 distinct): ['0', '6', '7', '4', '9']
contributionTrailerPolicies (uint8, 6 distinct): ['0', '2', '1', '3', '4', '5']
contributionTractorPolicies (uint8, 6 distinct): ['0', '3', '4', '5', '6', '7']
contributionAgriculturalMachinesPolicies (uint8, 6 distinct): ['0', '4', '3', '2', '6', '1']
contributionMopedPolicies (uint8, 6 distinct): ['0', '3', '4', '2', '5', '6']
contributionLifeInsurances (uint8, 10 distinct): ['0', '4', '3', '5', '6', '2', '1', '7', '9', '8']
contributionPrivateAccidentInsurancePolicies (uint8, 7 distinct): ['0', '2', '3', '1', '4', '5', '6']
contributionFamilyAccidentsInsurancePolicies (uint8, 3 distinct): ['0', '2', '3']
contributionDisabilityInsurancePolicies (uint8, 5 distinct): ['0', '6', '7', '5', '4']
contributionFirePolicies (uint8, 9 distinct): ['0', '4', '3', '2', '5', '6', '1', '7', '8']
contributionSurfboardPolicies (uint8, 4 distinct): ['0', '2', '1', '3']
contributionBoatPolicies (uint8, 7 distinct): ['0', '4', '2', '3', '1', '5', '6']
contributionBicyclePolicies (uint8, 2 distinct): ['0', '1']
contributionPropertyInsurancePolicies (uint8, 7 distinct): ['0', '1', '2', '4', '3', '6', '5']
contributionSocialSecurityInsurancePolicies (uint8, 5 distinct): ['0', '4', '3', '2', '5']
numberOfPrivateThirdPartyInsurance (uint8, 3 distinct): ['0', '1', '2']
numberOfThirdPartyInsuranceFirms (uint8, 3 distinct): ['0', '1', '5']
numberOfThirdPartyInsuranceAgriculture (uint8, 2 distinct): ['0', '1']
numberOfCarPolicies (uint8, 9 distinct): ['0', '1', '2', '3', '4', '6', '5', '12', '7']
numberOfDeliveryVanPolicies (uint8, 6 distinct): ['0', '1', '2', '3', '5', '4']
numberOfMotorcycleScooterPolicies (uint8, 5 distinct): ['0', '1', '2', '3', '8']
numberOfLorryPolicies (uint8, 5 distinct): ['0', '1', '2', '4', '3']
numberOfTrailerPolicies (uint8, 4 distinct): ['0', '1', '2', '3']
numberOfTractorPolicies (uint8, 7 distinct): ['0', '1', '2', '3', '4', '6', '5']
numberOfAgriculturalMachinesPolicies (uint8, 6 distinct): ['0', '1', '2', '3', '4', '6']
numberOfMopedPolicies (uint8, 4 distinct): ['0', '1', '2', '3']
numberOfLifeInsurances (uint8, 7 distinct): ['0', '1', '2', '3', '4', '5', '8']
numberOfPrivateAccidentInsurancePolicies (uint8, 2 distinct): ['0', '1']
numberOfFamilyAccidentsInsurancePolicies (uint8, 2 distinct): ['0', '1']
numberOfDisabilityInsurancePolicies (uint8, 3 distinct): ['0', '1', '2']
numberOfFirePolicies (uint8, 8 distinct): ['1', '0', '2', '3', '4', '5', '6', '7']
numberOfSurfboardPolicies (uint8, 2 distinct): ['0', '1']
numberOfBoatPolicies (uint8, 3 distinct): ['0', '1', '2']
numberOfBicyclePolicies (uint8, 5 distinct): ['0', '1', '2', '3', '4']
numberOfPropertyInsurancePolicies (uint8, 3 distinct): ['0', '1', '2']
numberOfSocialSecurityInsurancePolicies (uint8, 3 distinct): ['0', '1', '2']
'''

CONTEXT = "Mobile Home Insurance Policy"
TARGET = CuratedTarget(raw_name="MobileHomePolicy", task_type=SupervisedTask.BINARY)
