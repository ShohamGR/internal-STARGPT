from typing import Any, Optional

import numpy as np

from tabstar_paper.datasets.curation_objects import CuratedTarget, CuratedFeature
from tabstar_paper.datasets.objects import SupervisedTask

'''
Dataset Name: sukritchatterjee/used-cars-dataset-cardekho/cars_details_merges.csv
====
Examples: 37814
====
URL: https://www.kaggle.com/sukritchatterjee/used-cars-dataset-cardekho/cars_details_merges.csv
====
Description: 
Used Cars Dataset (CarDekho)
A dataset of used cars with all of their details and listing price.

About Dataset
This dataset contains information about ~38000 cars listed on Cardekho.

There are three CSV files in this dataset -

cars_overview.csv : Overview of the cars, contains basic info about the cars such as transmission type, location and the listing price.
car_details.csv : This file contains the almost all the cars in the overview file along with many other details, such as the features of the cars, the type of owner, etc.
car_details_merges.csv : This file is the merged version of the above two files, contains the basic as well as the detailed information of all the cars.
feature_dictionary.csv : Since the data is quite big, this file explains what information each column in the dataset has.

Points to note about the data:

The dataset contains columns which can have duplicate information since the data is scrapped using an API. It is advised to clean the data before using it.
There are multiple unique identifiers for each car, but using usedCarSkuId is recommended.
The price of the cars is under the column named price which has values such as ₹ 3.50 Lakh. We also have another column indicating the price in a continuous variable type called pu
Disclaimer
This data is meant for academic and research purposes and should not be used for commercial purposes.

====
Target Variable: pu (object, 6865 distinct): ['300000', '350000', '400000', '500000', '250000', '450000', '600000', '200000', '550000', '650000']
====
Features:

position (int64, 20 distinct): ['19', '20', '10', '14', '18', '11', '12', '16', '8', '15']
loc (object, 511 distinct): ['Pune City', 'Gurgaon', 'Bangalore City', 'New Delhi G.P.O.', 'pune city', 'gurgaon', 'new delhi g.p.o.', 'bangalore city', 'Mahadevapura', 'Noida']
myear (int64, 34 distinct): ['2017', '2018', '2014', '2015', '2016', '2019', '2013', '2021', '2020', '2012']
bt (object, 11 distinct): ['Hatchback', 'Sedan', 'SUV', 'MUV', 'Minivans', 'Luxury Vehicles', 'Pickup Trucks', 'Convertibles', 'Coupe', 'Wagon']
tt (object, 2 distinct): ['Manual', 'Automatic']
ft (object, 5 distinct): ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric']
km (object, 23863 distinct): ['70,000', '1,20,000', '80,000', '60,000', '90,000', '50,000', '40,000', '1,10,000', '1,00,000', '35,000']
ip (int64, 2 distinct): ['0', '1']
imgCount (int64, 54 distinct): ['15', '20', '21', '10', '11', '22', '1', '16', '12', '17']
threesixty (bool, 2 distinct): ['0', '1']
dvn (object, 4159 distinct): ['Maruti Swift VXI', 'Maruti Alto 800 LXI', 'Maruti Wagon R LXI CNG', 'Maruti Wagon R VXI BS IV', 'Maruti Swift VDI BSIV', 'Maruti Swift Dzire VDI', 'Honda City 1.5 S MT', 'Hyundai Grand i10 Sportz', 'Maruti Swift Dzire VXI', 'Hyundai i10 Magna']
oem (object, 46 distinct): ['Maruti', 'Hyundai', 'Honda', 'Mahindra', 'Tata', 'Toyota', 'Ford', 'Renault', 'Volkswagen', 'Skoda']
model (object, 382 distinct): ['Honda City', 'Hyundai i20', 'Maruti Swift', 'Maruti Wagon R', 'Maruti Swift Dzire', 'Hyundai i10', 'Hyundai Grand i10', 'Hyundai Creta', 'Hyundai Verna', 'Maruti Baleno']
modelId (int64, 682 distinct): ['627', '614', '586', '245', '262', '626', '255', '220', '992', '249']
vid (object, 4159 distinct): ['Maruti Swift VXI', 'Maruti Alto 800 LXI', 'Maruti Wagon R LXI CNG', 'Maruti Wagon R VXI BS IV', 'Maruti Swift VDI BSIV', 'Maruti Swift Dzire VDI', 'Honda City 1.5 S MT', 'Hyundai Grand i10 Sportz', 'Maruti Swift Dzire VXI', 'Hyundai i10 Magna']
centralVariantId (int64, 4585 distinct): ['4312', '1245', '4164', '7084', '4310', '1568', '3962', '1549', '3943', '1570']
variantName (object, 3488 distinct): ['VXI', 'LXI', 'Sportz', 'VDI', 'Magna', 'LXI CNG', 'VXI BS IV', 'VDI BSIV', 'Sportz 1.2', '1.5 S MT']
city_x (object, 617 distinct): ['New Delhi', 'Bangalore', 'Pune', 'Gurgaon', 'Mumbai', 'Hyderabad', 'Noida', 'Ahmedabad', 'Kolkata', 'Ghaziabad']
discountValue (int64, 30 distinct): ['0', '5000', '100000', '3000', '4000', '60000', '10000', '7000', '200000', '30000']
msp (int64, 3 distinct): ['0', '1093000', '959000']
priceSaving (object, 2 distinct): ['₹94K Regular Off', '₹60K Regular Off']
pageNo (int64, 162 distinct): ['2', '53', '54', '56', '55', '52', '62', '57', '61', '60']
utype (object, 2 distinct): ['Dealer', 'Individual']
views (int64, 2143 distinct): ['37', '47', '45', '39', '36', '38', '35', '22', '28', '25']
tmGaadiStore (bool, 2 distinct): ['0', '1']
emiwidget (object, 6206 distinct): ['{}', "{'title': 'EMI starts', 'cost': '12,411', 'duration': '/mo', 'caption': 'Interest calculated at 14.5% for 48 months', 'btntext': 'Check Your Loan', 'btnUrl': 'https://loan.cardekho.com/used-car?utm_source=cardekho&utm_medium=internal&utm_campaign=UsedCarsSRP_EMIstarts'}", "{'title': 'EMI starts', 'cost': '14,893', 'duration': '/mo', 'caption': 'Interest calculated at 14.5% for 48 months', 'btntext': 'Check Your Loan', 'btnUrl': 'https://loan.cardekho.com/used-car?utm_source=cardekho&utm_medium=internal&utm_campaign=UsedCarsSRP_EMIstarts'}", "{'title': 'EMI starts', 'cost': '16,134', 'duration': '/mo', 'caption': 'Interest calculated at 14.5% for 48 months', 'btntext': 'Check Your Loan', 'btnUrl': 'https://loan.cardekho.com/used-car?utm_source=cardekho&utm_medium=internal&utm_campaign=UsedCarsSRP_EMIstarts'}", "{'title': 'EMI starts', 'cost': '13,652', 'duration': '/mo', 'caption': 'Interest calculated at 14.5% for 48 months', 'btntext': 'Check Your Loan', 'btnUrl': 'https://loan.cardekho.com/used-car?utm_source=cardekho&utm_medium=internal&utm_campaign=UsedCarsSRP_EMIstarts'}", "{'title': 'EMI starts', 'cost': '17,375', 'duration': '/mo', 'caption': 'Interest calculated at 14.5% for 48 months', 'btntext': 'Check Your Loan', 'btnUrl': 'https://loan.cardekho.com/used-car?utm_source=cardekho&utm_medium=internal&utm_campaign=UsedCarsSRP_EMIstarts'}", "{'title': 'EMI starts', 'cost': '11,170', 'duration': '/mo', 'caption': 'Interest calculated at 14.5% for 48 months', 'btntext': 'Check Your Loan', 'btnUrl': 'https://loan.cardekho.com/used-car?utm_source=cardekho&utm_medium=internal&utm_campaign=UsedCarsSRP_EMIstarts'}", "{'title': 'EMI starts', 'cost': '18,616', 'duration': '/mo', 'caption': 'Interest calculated at 14.5% for 48 months', 'btntext': 'Check Your Loan', 'btnUrl': 'https://loan.cardekho.com/used-car?utm_source=cardekho&utm_medium=internal&utm_campaign=UsedCarsSRP_EMIstarts'}", "{'title': 'EMI starts', 'cost': '19,857', 'duration': '/mo', 'caption': 'Interest calculated at 14.5% for 48 months', 'btntext': 'Check Your Loan', 'btnUrl': 'https://loan.cardekho.com/used-car?utm_source=cardekho&utm_medium=internal&utm_campaign=UsedCarsSRP_EMIstarts'}", "{'title': 'EMI starts', 'cost': '9,929', 'duration': '/mo', 'caption': 'Interest calculated at 14.5% for 48 months', 'btntext': 'Check Your Loan', 'btnUrl': 'https://loan.cardekho.com/used-car?utm_source=cardekho&utm_medium=internal&utm_campaign=UsedCarsSRP_EMIstarts'}"]
transmissionType (object, 2 distinct): ['Manual', 'Automatic']
dynx_itemid2_x (int64, 4585 distinct): ['4312', '1245', '4164', '7084', '4310', '1568', '3962', '1549', '3943', '1570']
dynx_totalvalue_x (int64, 6865 distinct): ['300000', '350000', '400000', '500000', '250000', '450000', '600000', '200000', '550000', '650000']
leadForm (int64, 2 distinct): ['1', '0']
pageType (object, 2 distinct): ['cls', 'ucr']
carType (object, 3 distinct): ['partner', 'corporate', 'assured']
corporateId (int64, 6 distinct): ['7', '17', '13', '11', '16', '15']
top_features (object, 400 distinct): ["['Power Steering', 'Power Windows Front', 'Air Conditioner', 'Heater', 'Adjustable Head Lights', 'Fog Lights Front', 'Anti Lock Braking System', 'Centeral Locking', 'Cd Player']", "['Power Steering', 'Power Windows Front', 'Air Conditioner', 'Heater', 'Adjustable Head Lights', 'Fog Lights Front', 'Anti Lock Braking System', 'Centeral Locking', 'Radio']", "['Power Steering', 'Power Windows Front', 'Air Conditioner', 'Heater', 'Adjustable Head Lights', 'Fog Lights Front', 'Anti Lock Braking System', 'Brake Assist', 'Cd Player']", "['Power Steering', 'Power Windows Front', 'Air Conditioner', 'Heater', 'Adjustable Head Lights', 'Fog Lights Front', 'Anti Lock Braking System', 'Brake Assist', 'Radio']", "['Power Steering', 'Power Windows Front', 'Air Conditioner', 'Heater', 'Adjustable Head Lights', 'Power Adjustable Exterior Rear View Mirror', 'Anti Lock Braking System', 'Centeral Locking', 'Radio']", "['Power Steering', 'Power Windows Front', 'Air Conditioner', 'Heater', 'Adjustable Head Lights', 'Power Adjustable Exterior Rear View Mirror', 'Anti Lock Braking System', 'Brake Assist', 'Cd Player']", "['Power Steering', 'Power Windows Front', 'Air Conditioner', 'Heater', 'Adjustable Head Lights', 'Fog Lights Front', 'Centeral Locking', 'Power Door Locks', 'Cd Player']", "['Power Steering', 'Power Windows Front', 'Air Conditioner', 'Heater', 'Adjustable Head Lights', 'Power Adjustable Exterior Rear View Mirror', 'Anti Lock Braking System', 'Centeral Locking', 'Cd Player']", "['Power Steering', 'Power Windows Front', 'Air Conditioner', 'Heater', 'Adjustable Head Lights', 'Manually Adjustable Exterior Rear View Mirror', 'Centeral Locking', 'Child Safety Locks']", "['Power Steering', 'Power Windows Front', 'Air Conditioner', 'Heater', 'Adjustable Head Lights', 'Manually Adjustable Exterior Rear View Mirror', 'Anti Lock Braking System', 'Centeral Locking']"]
comfort_features (object, 2016 distinct): ["['Power Steering', 'Power Windows Front', 'Power Windows Rear', 'Remote Trunk Opener', 'Remote Fuel Lid Opener', 'Low Fuel Warning Light', 'Accessory Power Outlet', 'Vanity Mirror', 'Rear Seat Headrest', 'Cup Holders Front']", "['Power Steering', 'Power Windows Front', 'Power Windows Rear', 'Air Quality Control', 'Remote Trunk Opener', 'Remote Fuel Lid Opener', 'Low Fuel Warning Light', 'Accessory Power Outlet', 'Trunk Light', 'Vanity Mirror', 'Rear Reading Lamp', 'Rear Seat Headrest', 'Rear Seat Centre Arm Rest', 'Height Adjustable Front Seat Belts', 'Cup Holders Front', 'Cup Holders Rear', 'Seat Lumbar Support', 'Multifunction Steering Wheel', 'Cruise Control', 'Rear ACVents']", "['Power Steering', 'Power Windows Front', 'Power Windows Rear', 'Remote Fuel Lid Opener', 'Low Fuel Warning Light', 'Accessory Power Outlet', 'Vanity Mirror', 'Rear Seat Headrest', 'Cup Holders Front', 'Seat Lumbar Support']", "['Power Steering', 'Power Windows Front', 'Power Windows Rear', 'Air Quality Control', 'Remote Trunk Opener', 'Remote Fuel Lid Opener', 'Low Fuel Warning Light', 'Accessory Power Outlet', 'Trunk Light', 'Vanity Mirror', 'Rear Seat Headrest', 'Rear Seat Centre Arm Rest', 'Cup Holders Front', 'Cup Holders Rear', 'Multifunction Steering Wheel', 'Cruise Control', 'Rear ACVents']", "['Power Steering', 'Power Windows Front', 'Remote Trunk Opener', 'Remote Fuel Lid Opener', 'Low Fuel Warning Light', 'Accessory Power Outlet', 'Rear Seat Headrest', 'Cup Holders Front']", "['Power Steering', 'Power Windows Front', 'Power Windows Rear', 'Remote Trunk Opener', 'Remote Fuel Lid Opener', 'Low Fuel Warning Light', 'Accessory Power Outlet', 'Trunk Light', 'Vanity Mirror', 'Rear Reading Lamp', 'Rear Seat Headrest', 'Rear Seat Centre Arm Rest', 'Cup Holders Front', 'Cup Holders Rear', 'Multifunction Steering Wheel']", "['Power Steering', 'Power Windows Front', 'Power Windows Rear', 'Remote Trunk Opener', 'Remote Fuel Lid Opener', 'Low Fuel Warning Light', 'Accessory Power Outlet', 'Trunk Light', 'Vanity Mirror', 'Rear Reading Lamp', 'Rear Seat Headrest', 'Rear Seat Centre Arm Rest', 'Height Adjustable Front Seat Belts', 'Cup Holders Front', 'Cup Holders Rear', 'Seat Lumbar Support', 'Multifunction Steering Wheel']", "['Power Steering', 'Power Windows Front', 'Power Windows Rear', 'Remote Trunk Opener', 'Remote Fuel Lid Opener', 'Low Fuel Warning Light', 'Accessory Power Outlet', 'Vanity Mirror', 'Rear Seat Headrest', 'Cup Holders Front', 'Multifunction Steering Wheel']", "['Power Steering', 'Power Windows Front', 'Low Fuel Warning Light', 'Rear Seat Headrest', 'Cup Holders Front']", '[]']
interior_features (object, 527 distinct): ["['Air Conditioner', 'Heater', 'Adjustable Steering', 'Digital Odometer', 'Tachometer', 'Electronic Multi Tripmeter', 'Fabric Upholstery', 'Glove Compartment', 'Digital Clock']", "['Air Conditioner', 'Heater', 'Adjustable Steering', 'Digital Odometer', 'Tachometer', 'Electronic Multi Tripmeter', 'Fabric Upholstery', 'Glove Compartment', 'Digital Clock', 'Height Adjustable Driver Seat']", "['Air Conditioner', 'Heater', 'Adjustable Steering', 'Digital Odometer', 'Tachometer', 'Electronic Multi Tripmeter', 'Fabric Upholstery', 'Glove Compartment', 'Digital Clock', 'Outside Temperature Display', 'Height Adjustable Driver Seat']", "['Air Conditioner', 'Heater', 'Adjustable Steering', 'Digital Odometer', 'Tachometer', 'Electronic Multi Tripmeter', 'Fabric Upholstery', 'Leather Steering Wheel', 'Glove Compartment', 'Digital Clock', 'Outside Temperature Display', 'Height Adjustable Driver Seat']", "['Air Conditioner', 'Heater', 'Digital Odometer', 'Tachometer', 'Electronic Multi Tripmeter', 'Fabric Upholstery', 'Glove Compartment', 'Digital Clock']", "['Air Conditioner', 'Heater', 'Adjustable Steering', 'Digital Odometer', 'Tachometer', 'Electronic Multi Tripmeter', 'Fabric Upholstery', 'Leather Steering Wheel', 'Glove Compartment', 'Digital Clock', 'Height Adjustable Driver Seat']", "['Air Conditioner', 'Heater', 'Adjustable Steering', 'Digital Odometer', 'Tachometer', 'Electronic Multi Tripmeter', 'Fabric Upholstery', 'Glove Compartment', 'Digital Clock', 'Outside Temperature Display']", "['Air Conditioner', 'Heater', 'Adjustable Steering', 'Digital Odometer', 'Tachometer', 'Electronic Multi Tripmeter', 'Leather Seats', 'Leather Steering Wheel', 'Glove Compartment', 'Digital Clock', 'Outside Temperature Display', 'Height Adjustable Driver Seat']", "['Air Conditioner', 'Heater', 'Adjustable Steering', 'Digital Odometer', 'Tachometer', 'Electronic Multi Tripmeter', 'Fabric Upholstery', 'Glove Compartment', 'Digital Clock', 'Cigarette Lighter']", "['Air Conditioner', 'Heater', 'Adjustable Steering', 'Digital Odometer', 'Tachometer', 'Electronic Multi Tripmeter', 'Fabric Upholstery', 'Glove Compartment', 'Digital Clock', 'Height Adjustable Driver Seat', 'Dual Tone Dashboard']"]
exterior_features (object, 1893 distinct): ["['Adjustable Head Lights', 'Fog Lights Front', 'Power Adjustable Exterior Rear View Mirror', 'Electric Folding Rear View Mirror', 'Rear Window Wiper', 'Rear Window Washer', 'Rear Window Defogger', 'Alloy Wheels', 'Integrated Antenna', 'Tinted Glass', 'Rear Spoiler', 'Outside Rear View Mirror Turn Indicators', 'Chrome Grille', 'Chrome Garnish', 'Smoke Headlamps', 'Roof Rail']", "['Adjustable Head Lights', 'Fog Lights Front', 'Power Adjustable Exterior Rear View Mirror', 'Electric Folding Rear View Mirror', 'Rear Window Defogger', 'Alloy Wheels', 'Power Antenna', 'Outside Rear View Mirror Turn Indicators', 'Chrome Grille', 'Chrome Garnish']", '[]', "['Adjustable Head Lights', 'Fog Lights Front', 'Fog Lights Rear', 'Power Adjustable Exterior Rear View Mirror', 'Wheel Covers', 'Power Antenna', 'Tinted Glass', 'Outside Rear View Mirror Turn Indicators']", "['Adjustable Head Lights', 'Manually Adjustable Exterior Rear View Mirror', 'Power Antenna', 'Chrome Grille']", "['Adjustable Head Lights', 'Manually Adjustable Exterior Rear View Mirror', 'Wheel Covers', 'Halogen Headlamps']", "['Adjustable Head Lights', 'Manually Adjustable Exterior Rear View Mirror', 'Power Antenna', 'Chrome Grille', 'Chrome Garnish', 'Roof Rail']", "['Adjustable Head Lights', 'Manually Adjustable Exterior Rear View Mirror', 'Wheel Covers', 'Power Antenna', 'Chrome Grille', 'Halogen Headlamps']", "['Adjustable Head Lights', 'Manually Adjustable Exterior Rear View Mirror', 'Wheel Covers', 'Power Antenna']", "['Adjustable Head Lights', 'Manually Adjustable Exterior Rear View Mirror', 'Integrated Antenna', 'Tinted Glass']"]
safety_features (object, 2116 distinct): ["['Anti Lock Braking System', 'Centeral Locking', 'Power Door Locks', 'Child Safety Locks', 'Anti Theft Alarm', 'Driver Air Bag', 'Passenger Air Bag', 'Day Night Rear View Mirror', 'Passenger Side Rear View Mirror', 'Halogen Headlamps', 'Rear Seat Belts', 'Seat Belt Warning', 'Door Ajar Warning', 'Side Impact Beams', 'Front Impact Beams', 'Adjustable Seats', 'Keyless Entry', 'Centrally Mounted Fuel Tank', 'Engine Immobilizer', 'Crash Sensor', 'Ebd', 'Rear Camera', 'Anti Theft Device']", "['Centeral Locking', 'Power Door Locks', 'Child Safety Locks', 'Day Night Rear View Mirror', 'Passenger Side Rear View Mirror', 'Halogen Headlamps', 'Rear Seat Belts', 'Seat Belt Warning', 'Door Ajar Warning', 'Side Impact Beams', 'Front Impact Beams', 'Adjustable Seats', 'Keyless Entry', 'Centrally Mounted Fuel Tank', 'Engine Immobilizer', 'Anti Theft Device']", "['Centeral Locking', 'Power Door Locks', 'Child Safety Locks', 'Anti Theft Alarm', 'Day Night Rear View Mirror', 'Passenger Side Rear View Mirror', 'Halogen Headlamps', 'Rear Seat Belts', 'Seat Belt Warning', 'Door Ajar Warning', 'Side Impact Beams', 'Front Impact Beams', 'Adjustable Seats', 'Keyless Entry', 'Centrally Mounted Fuel Tank', 'Engine Immobilizer', 'Anti Theft Device']", "['Anti Lock Braking System', 'Brake Assist', 'Centeral Locking', 'Power Door Locks', 'Child Safety Locks', 'Anti Theft Alarm', 'Driver Air Bag', 'Passenger Air Bag', 'Day Night Rear View Mirror', 'Passenger Side Rear View Mirror', 'Halogen Headlamps', 'Rear Seat Belts', 'Seat Belt Warning', 'Door Ajar Warning', 'Side Impact Beams', 'Front Impact Beams', 'Adjustable Seats', 'Keyless Entry', 'Engine Immobilizer', 'Engine Check Warning']", "['Anti Lock Braking System', 'Centeral Locking', 'Power Door Locks', 'Child Safety Locks', 'Anti Theft Alarm', 'Driver Air Bag', 'Passenger Air Bag', 'Day Night Rear View Mirror', 'Passenger Side Rear View Mirror', 'Halogen Headlamps', 'Rear Seat Belts', 'Seat Belt Warning', 'Door Ajar Warning', 'Side Impact Beams', 'Front Impact Beams', 'Adjustable Seats', 'Keyless Entry', 'Centrally Mounted Fuel Tank', 'Engine Immobilizer', 'Crash Sensor', 'Ebd', 'Anti Theft Device']", '[]', "['Anti Lock Braking System', 'Centeral Locking', 'Power Door Locks', 'Child Safety Locks', 'Driver Air Bag', 'Passenger Air Bag', 'Day Night Rear View Mirror', 'Passenger Side Rear View Mirror', 'Halogen Headlamps', 'Rear Seat Belts', 'Seat Belt Warning', 'Door Ajar Warning', 'Side Impact Beams', 'Front Impact Beams', 'Adjustable Seats', 'Keyless Entry', 'Centrally Mounted Fuel Tank', 'Engine Immobilizer', 'Crash Sensor', 'Ebd', 'Follow Me Home Headlamps', 'Rear Camera', 'Anti Theft Device', 'Impact Sensing Auto Door Lock']", "['Centeral Locking', 'Day Night Rear View Mirror', 'Passenger Side Rear View Mirror', 'Halogen Headlamps', 'Rear Seat Belts', 'Side Impact Beams', 'Front Impact Beams', 'Adjustable Seats', 'Centrally Mounted Fuel Tank', 'Engine Immobilizer']", "['Anti Lock Braking System', 'Centeral Locking', 'Power Door Locks', 'Child Safety Locks', 'Anti Theft Alarm', 'Driver Air Bag', 'Passenger Air Bag', 'Passenger Side Rear View Mirror', 'Halogen Headlamps', 'Rear Seat Belts', 'Seat Belt Warning', 'Door Ajar Warning', 'Side Impact Beams', 'Front Impact Beams', 'Adjustable Seats', 'Keyless Entry', 'Centrally Mounted Fuel Tank', 'Engine Immobilizer', 'Engine Check Warning', 'Crash Sensor', 'Ebd', 'Anti Theft Device', 'Isofix Child Seat Mounts', 'Pretensioners And Force Limiter Seatbelts', 'No Of Airbags']", "['Centeral Locking', 'Child Safety Locks', 'Day Night Rear View Mirror', 'Passenger Side Rear View Mirror', 'Halogen Headlamps', 'Rear Seat Belts', 'Seat Belt Warning', 'Side Impact Beams', 'Front Impact Beams', 'Adjustable Seats', 'Centrally Mounted Fuel Tank', 'Engine Immobilizer', 'Engine Check Warning', 'Anti Theft Device']"]
Color (object, 798 distinct): ['White', 'Silver', 'Grey', 'Red', 'Blue', 'Black', 'Brown', 'Other', 'Golden', 'Gray']
Engine Type (object, 658 distinct): ['In-Line Engine', 'Kappa VTVT Petrol Engine', 'Petrol Engine', 'DDiS Diesel Engine', 'K Series Petrol Engine', 'Diesel Engine', 'i-VTEC Petrol Engine', 'mHawk Diesel Engine', 'VTVT Petrol Engine', 'i VTEC Engine']
Displacement (float64, 187 distinct): ['1197.0', '1248.0', '998.0', '1497.0', '1498.0', '1199.0', '1198.0', '2179.0', '999.0', '1086.0']
Max Power (object, 1018 distinct): ['81.80bhp@6000rpm', '88.5bhp@4000rpm', '74bhp@4000rpm', '81.86bhp@6000rpm', '98.6bhp@3600rpm', '140bhp@3750rpm', '78.9bhp@6000rpm', '117.3bhp@6600rpm', '82bhp@6000rpm', '121.3bhp@6400rpm']
Max Torque (object, 844 distinct): ['200Nm@1750rpm', '90Nm@3500rpm', '113Nm@4200rpm', '114Nm@4000rpm', '190Nm@2000rpm', '145Nm@4600rpm', '110Nm@4800rpm', '109Nm@4500rpm', '115Nm@4000rpm', '69Nm@3500rpm']
No of Cylinder (float64, 11 distinct): ['4.0', '3.0', '6.0', '5.0', '2.0', '7.0', '8.0', '1.0', '12.0', '10.0']
Values per Cylinder (float64, 7 distinct): ['4.0', '2.0', '3.0', '5.0', '1.0', '48.0', '8.0']
Value Configuration (object, 13 distinct): ['DOHC', 'SOHC', 'DOHC ', 'undefined', 'iDSI', 'DOHC with VIS', 'DOHC with VGT', '16 Modules 48 Cells', 'DOHC with TIS', '16-valve DOHC layout']
BoreX Stroke (object, 224 distinct): ['69.6 X 82 mm', '73.0 X 89.4 mm', '69 x 72 mm', '71 x 75.6 mm', '69.6 X 82', '73 X 82 mm', '73 X 71.5 mm', '73 x 72 mm', '77 X 85.8 mm', '73.5 X 88.3 mm']
Turbo Charger (object, 9 distinct): ['No', 'Yes', 'Twin', 'YES', 'NO', 'no', 'yes', 'twin', 'Turbo']
Super Charger (object, 5 distinct): ['No', 'Yes', 'NO', 'yes', 'no']
Length (mm) (float64, 411 distinct): ['3995.0', '4440.0', '3765.0', '3985.0', '3585.0', '4585.0', '3840.0', '4490.0', '4315.0', '4270.0']
Width (mm) (float64, 276 distinct): ['1695.0', '1735.0', '1680.0', '1595.0', '1660.0', '1790.0', '1495.0', '1734.0', '1475.0', '1745.0']
Height (mm) (float64, 300 distinct): ['1505.0', '1520.0', '1530.0', '1475.0', '1495.0', '1550.0', '1700.0', '1640.0', '1485.0', '1510.0']
Wheel Base (mm) (float64, 219 distinct): ['2450.0', '2600.0', '2380.0', '2425.0', '2400.0', '2500.0', '2360.0', '2570.0', '2520.0', '2700.0']
Front Tread (mm) (float64, 156 distinct): ['1295.0', '1505.0', '1480.0', '1400.0', '1479.0', '1530.0', '1485.0', '1470.0', '1495.0', '1560.0']
Rear Tread (mm) (float64, 172 distinct): ['1290.0', '1503.0', '1493.0', '1465.0', '1385.0', '1495.0', '1530.0', '1520.0', '1480.0', '1505.0']
Kerb Weight (object, 831 distinct): ['1066kg', '935kg', '885kg', '870kg', '860kg', '960kg', '1100kg', '1060kg', '855-885', '1025kg']
Gross Weight (object, 436 distinct): ['1350kg', '1340kg', '2510kg', '1680kg', '1185kg', '1415kg', '1505kg', '1315kg', '2450kg', '1490kg']
Gear Box (object, 132 distinct): ['5 Speed', '6 Speed', '5-Speed', '7 Speed', '5 Speed ', '6-Speed', '6 Speed ', '8 Speed', '4 Speed', '5']
Drive Type (object, 24 distinct): ['FWD', 'RWD', 'AWD', '2WD', '4WD', '2 WD', '4X2', 'FWD ', '4X4', 'Front Wheel Drive']
Seating Capacity (float64, 11 distinct): ['5.0', '7.0', '8.0', '4.0', '6.0', '9.0', '2.0', '10.0', '0.0', '13.0']
Steering Type (object, 11 distinct): ['Power', 'Electric', 'Manual', 'Electronic', 'Electrical', 'power', 'EPAS', 'Hydraulic', 'electric', 'MT']
Turning Radius (object, 239 distinct): ['5.3 metres', '5.2 metres', '4.8 metres', '4.7 metres', '4.6 metres', '5.4 metres', '5.6 metres', '4.9 meters', '4.9 metres', '4.5 metres']
Front Brake Type (object, 43 distinct): ['Disc', 'Ventilated Disc', 'Disc ', 'Solid Disc', 'Ventilated Discs', 'Disc & Caliper Type', 'Disk', 'Ventilated Disc ', 'Ventilated Disk', 'Ventilated discs']
Rear Brake Type (object, 48 distinct): ['Drum', 'Disc', 'Ventilated Disc', 'Solid Disc', 'Self-Adjusting Drum', 'Discs', 'Disc & Caliper Type', 'Leading-Trailing Drum', 'Ventilated Discs', 'Leading & Trailing Drum']
Top Speed (object, 372 distinct): ['165 Kmph', '170 Kmph', '180 Kmph', '160 Kmph', '195 Kmph', '190 Kmph', '172 kmph', '190 kmph', '160 kmph', '145 Kmph']
Acceleration (object, 414 distinct): ['10 Seconds', '15 Seconds', '14 Seconds', '19 Seconds', '12.9 Seconds', '18.6 Seconds', '13.3 Seconds', '12.36 seconds', '13.2 Seconds', '14.3 Seconds']
Tyre Type (object, 37 distinct): ['Tubeless,Radial', 'Tubeless', 'Tubeless, Radial', 'Tubeless Tyres', 'Radial', 'Tubeless,Radial ', 'Radial, Tubeless', 'Tubeless Radial Tyres', 'Radial, Tubless', 'Tubeless Tyres, Radial']
No Door Numbers (float64, 5 distinct): ['5.0', '4.0', '3.0', '2.0', '6.0']
Cargo Volumn (object, 398 distinct): ['510-litres', '400-litres', '339-litres', '256-liters', '350', '475-litres', '295-litres', '180-liters', '460-litres', '328-litres']
originalLocation (float64, 0 distinct): []
page_title (object, 37814 distinct): ['Used Tata Tigor 1.05 Revotorq XZ Car in Pune, 2018 Model (Id- a96fbcd7-c183-4829-ae97-b2581afe4bac) - Find Best Deals! | CarDekho.com', 'Used Maruti Wagon R LXI CNG Car in Lucknow, 2016 Model (Id- 7111bf25-97af-47f9-867b-40879190d800) - Find Best Deals! | CarDekho.com', 'Used Maruti Celerio Green VXI Car in Mumbai, 2015 Model (Id- c309efc1-efaf-4f82-81ad-dcb38eb36665) - Find Best Deals! | CarDekho.com', 'Used Honda Amaze S Plus I-VTEC Car in New Delhi, 2015 Model (Id- 7609f710-0c97-4f00-9a47-9b9284b62d3a) - Find Best Deals! | CarDekho.com', 'Used Maruti Wagon R LXI CNG Car in New Delhi, 2013 Model (Id- 278b76e3-5539-4a5e-ae3e-353a2e3b6d7d) - Find Best Deals! | CarDekho.com', 'Used Maruti Ertiga VXI CNG Car in Mumbai, 2022 Model (Id- b1eab99b-a606-48dd-a75b-57feb8a9ad92) - Find Best Deals! | CarDekho.com', 'Used Maruti Wagon R LXI CNG Car in New Delhi, 2012 Model (Id- e030b70c-56b9-44b7-8b40-919fa92b6bbc) - Find Best Deals! | CarDekho.com', 'Used Maruti Alto Green LXi (CNG) Car in New Delhi, 2010 Model (Id- 6ca9c00f-b755-44e0-8680-a2d033346630) - Find Best Deals! | CarDekho.com', 'Used Hyundai Grand I10 1.2 Kappa Magna CNG BSIV Car in New Delhi, 2017 Model (Id- 27f36089-ecf1-415d-9401-f90e83813585) - Find Best Deals! | CarDekho.com', 'Used Maruti Wagon R CNG LXI Car in Gurgaon, 2021 Model (Id- 4ebf53b7-8910-43c3-947c-4feb145cb9c7) - Find Best Deals! | CarDekho.com']
seller_type_new (object, 2 distinct): ['dealer', 'individual']
seating_capacity_new (float64, 10 distinct): ['5.0', '7.0', '8.0', '4.0', '6.0', '9.0', '2.0', '10.0', '13.0', '14.0']
transmission_type (object, 2 distinct): ['manual', 'automatic']
model_year_new (int64, 34 distinct): ['2017', '2018', '2014', '2015', '2016', '2019', '2013', '2021', '2020', '2012']
model_name (object, 382 distinct): ['honda city', 'hyundai i20', 'maruti swift', 'maruti wagon r', 'maruti swift dzire', 'hyundai i10', 'hyundai grand i10', 'hyundai creta', 'hyundai verna', 'maruti baleno']
model_id_new (int64, 382 distinct): ['125', '148', '338', '344', '339', '146', '143', '138', '161', '322']
oem_name (object, 46 distinct): ['maruti', 'hyundai', 'honda', 'mahindra', 'tata', 'toyota', 'ford', 'renault', 'volkswagen', 'skoda']
state (object, 33 distinct): ['maharashtra', 'karnataka', 'delhi', 'uttar pradesh', 'haryana', 'gujarat', 'telangana', 'west bengal', 'tamil nadu', 'rajasthan']
city_id_new (int64, 617 distinct): ['49', '105', '205', '74', '201', '8', '348', '51', '338', '349']
fuel_type (object, 5 distinct): ['petrol', 'diesel', 'cng', 'lpg', 'electric']
max_engine_capacity_new (float64, 187 distinct): ['1197.0', '1248.0', '998.0', '1497.0', '1498.0', '1199.0', '1198.0', '2179.0', '999.0', '1086.0']
transmission_type_new (object, 2 distinct): ['manual', 'automatic']
km_driven (int64, 23862 distinct): ['70000', '120000', '80000', '60000', '90000', '50000', '40000', '110000', '100000', '35000']
model_new (object, 382 distinct): ['Honda City', 'Hyundai i20', 'Maruti Swift', 'Maruti Wagon R', 'Maruti Swift Dzire', 'Hyundai i10', 'Hyundai Grand i10', 'Hyundai Creta', 'Hyundai Verna', 'Maruti Baleno']
brand_name (object, 46 distinct): ['maruti', 'hyundai', 'honda', 'mahindra', 'tata', 'toyota', 'ford', 'renault', 'volkswagen', 'skoda']
engine_cc (object, 6 distinct): ['1000cc-2000cc', '500cc-1000cc', '2000cc-3000cc', '3000cc-4000cc', '4000cc-5000cc', '5000cc Plus']
fuel_type_new (object, 5 distinct): ['petrol', 'diesel', 'cng', 'lpg', 'electric']
car_segment (object, 11 distinct): ['Hatchback', 'Sedan', 'SUV', 'MUV', 'Minivans', 'Luxury Vehicles', 'Pickup Trucks', 'Convertibles', 'Coupe', 'Wagon']
city_name_new (object, 617 distinct): ['new delhi', 'bangalore', 'pune', 'gurgaon', 'mumbai', 'hyderabad', 'noida', 'ahmedabad', 'kolkata', 'ghaziabad']
city_y (object, 617 distinct): ['new delhi', 'bangalore', 'pune', 'gurgaon', 'mumbai', 'hyderabad', 'noida', 'ahmedabad', 'kolkata', 'ghaziabad']
engine_capacity_new (object, 6 distinct): ['1000cc-2000cc', '500cc-1000cc', '2000cc-3000cc', '3000cc-4000cc', '4000cc-5000cc', '5000cc Plus']
body_type_new (object, 12 distinct): ['Hatchback cars', 'Sedan cars', 'SUV cars', 'MUV cars', 'Minivans cars', 'Luxury Vehicles cars', 'Pickup Trucks cars', 'Convertibles cars', 'Coupe cars', ' cars']
owner_type_new (object, 6 distinct): ['first', 'second', 'third', 'fourth', 'fifth', 'unregistered car']
mileage_new (object, 627 distinct): ['18.9 kmpl', '17 kmpl', '18.6 kmpl', '18 kmpl', '20.36 kmpl', '24.3 kmpl', '16 kmpl', '16.8 kmpl', '21.21 kmpl', '15.1 kmpl']
model_year (int64, 34 distinct): ['2017', '2018', '2014', '2015', '2016', '2019', '2013', '2021', '2020', '2012']
variant_name (object, 4131 distinct): ['maruti swift vxi', 'maruti alto 800 lxi', 'maruti wagon r lxi cng', 'maruti wagon r vxi bs iv', 'maruti swift dzire vxi', 'maruti swift dzire vdi', 'maruti swift vdi bsiv', 'honda city 1.5 s mt', 'hyundai grand i10 sportz', 'hyundai i10 magna']
dynx_itemid2_y (int64, 4585 distinct): ['4312', '1245', '4164', '7084', '4310', '1568', '3962', '1549', '3943', '1570']
dynx_totalvalue_y (int64, 6867 distinct): ['300000', '350000', '400000', '500000', '250000', '450000', '600000', '200000', '550000', '650000']
brand_new (object, 46 distinct): ['maruti', 'hyundai', 'honda', 'mahindra', 'tata', 'toyota', 'ford', 'renault', 'volkswagen', 'skoda']
variant_new (object, 4131 distinct): ['maruti swift vxi', 'maruti alto 800 lxi', 'maruti wagon r lxi cng', 'maruti wagon r vxi bs iv', 'maruti swift dzire vxi', 'maruti swift dzire vdi', 'maruti swift vdi bsiv', 'honda city 1.5 s mt', 'hyundai grand i10 sportz', 'hyundai i10 magna']
exterior_color (object, 798 distinct): ['White', 'Silver', 'Grey', 'Red', 'Blue', 'Black', 'Brown', 'Other', 'Golden', 'Gray']
min_engine_capacity_new (float64, 187 distinct): ['1197.0', '1248.0', '998.0', '1497.0', '1498.0', '1199.0', '1198.0', '2179.0', '999.0', '1086.0']
owner_type (object, 6 distinct): ['first', 'second', 'third', 'fourth', 'fifth', 'unregistered car']
template_name_new (object, 4 distinct): ['used cardetail v2', 'used cardetail v2/corporate/13', 'used cardetail v2/corporate/16', 'used cardetail v2/ucr']
Fuel Suppy System (object, 99 distinct): ['MPFI', 'CRDi', 'CRDI', 'MPFi', 'Direct Injection', 'PGM-Fi', 'PGM - Fi', 'Common Rail', 'GDi', 'EFI(Electronic Fuel Injection)']
Compression Ratio (object, 100 distinct): ['17.6:1', '10.5:1', '10.3:1', '11.0:1', '16.0:1', '10.1:1', '16.5:1', '10.0:1', '9.0:1', '17.3:1']
Alloy Wheel Size (object, 18 distinct): ['16', '15', '14', '17', '13', '18', 'R16', '12', 'R17', '19']
Ground Clearance Unladen (mm) (float64, 26 distinct): ['190.0', '209.0', '178.0', '185.0', '192.0', '180.0', '120.0', '155.0', '170.0', '160.0']
'''

def remove_commas(text: str) -> str:
    return text.replace(',', '')

def remove_mm_unit(text: Any) -> Optional[float]:
    if isinstance(text, str):
        text = text.strip()
        if '-' in text:
            text = text.split('-')[0]
        for bad_char in ['`', ',']:
            text = text.replace(bad_char, '')
        if text.endswith('mm'):
            text = text[:-2]
        if text.endswith('m'):
            text = text[:-1]
        text = text.strip()
        if not text:
            return None
    if text is np.nan:
        return None
    if text in {'15t2'}:
        return None
    unit = float(text)
    return unit

CONTEXT = "User cars and listing price in the website Cardekho"
TARGET = CuratedTarget(raw_name="pu", task_type=SupervisedTask.REGRESSION, processing_func=remove_commas)
COLS_TO_DROP = ['price', 'price_segment_new', 'price_segment', 'vlink', "price_range_segment", "pi",
                # IDs
                "usedCarSkuId", "ucid", "sid", "dealer_id_new", "dealer_id", "used_carid", "dynx_itemid_x", "usedCarId",
                "dynx_itemid_y",
                # Constant columns
                "page_template", "template_Type_new", "experiment", "dynx_event", "dynx_pagetype", "vehicle_type_new",
                "page_type", "leadFormCta", "offers", "compare", "brandingIcon", "model_type_new", "car_type_new",
                "compare_car_details",
                # Image non-working URLs
                "images",]

MM_FEATURES = [CuratedFeature(raw_name=mmf, new_name=f"{mmf} (mm)", processing_func=remove_mm_unit)
               for mmf in ["Wheel Base", "Front Tread", "Rear Tread", "Length", "Width", "Height",
                           "Ground Clearance Unladen"]]

FEATURES = [] + MM_FEATURES
