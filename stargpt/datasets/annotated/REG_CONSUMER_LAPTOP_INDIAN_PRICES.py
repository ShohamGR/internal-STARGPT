from os.path import join

from pandas import DataFrame, read_csv

from tabstar_paper.datasets.curation_objects import CuratedTarget
from tabstar_paper.datasets.objects import SupervisedTask

'''
Dataset Name: dhanushbommavaram/laptop-dataset/
====
Examples: 984
====
URL: https://www.kaggle.com/dhanushbommavaram/laptop-dataset
====
Lennart:
This dataset contains laptop listings with specifications, textual descriptions, and target prices. Inputs include categorical
traits (e.g., Processor Brand), numerical specs (e.g., RAM, Screen Size), and verbose text (e.g., Sales Package, Additional
Features). Rather than filtering, all features are retained to simulate real-world noise and redundancy. It serves as a
robust stress test for tabular foundation models under messy conditions.
====
Description: 
The dataset is the laptop listings on Flipkart, an ecommerce website in India. This is an unprocessed dataset. It contains 984 unique entries. This unprocessed dataset contains 96 features (Ex: Laptop name, link to buy, price in Indian rupees, processor, GPU, etc.). This dataset was created by scraping data from the web (dated June).
====
Target Variable: Price (float64, 433 distinct): ['40990.0', '38990.0', '54990.0', '64990.0', '58990.0', '42990.0', '79990.0', '65990.0', '49990.0', '34990.0']
====
Features:

name (object, 943 distinct): ['DELL Ryzen 5 Hexa Core 5600H - (8 GB/512 GB SSD/Windows 11 Home/4 GB Graphics/NVIDIA GeForce RTX 3050/120 Hz) G15-5515 Gaming Laptop  (15.6 inch, Grey, 2.57 kg, With MS Office)', 'Avita SATUS Celeron Dual Core - (4 GB/128 GB SSD/Windows 11 Home) NU14A1INC43PN-MB Laptop  (14.1 Inch, Matt Black)', 'Lenovo IdeaPad 3 Core i3 10th Gen - (8 GB/256 GB SSD/Windows 10 Home) 15IIL05 Laptop  (15.6 inch, Platinum Grey, 1.85 kg, With MS Office)', 'ASUS ROG Flow X13 (2021) Ryzen 9 Octa Core Ryzen 9 5900HS 5th Gen - (32 GB/1 TB SSD/Windows 10 Home/4 GB Graphics/NVIDIA GeForce GTX 1650/120 Hz) GV301QH-K6461TS 2 in 1 Gaming Laptop  (13.4 Inch, Black, 1.3 KG, With MS Office)', 'ASUS ROG Strix SCAR 17 Core i9 12th Gen - (32 GB/1 TB SSD/Windows 11 Home/8 GB Graphics/NVIDIA GeForce RTX 3070 Ti) G733ZW-LL139WS Gaming Laptop  (17.3 inch, Off Black, 2.90 kg, With MS Office)', 'acer Aspire 7 Ryzen 5 Quad Core 3550H - (8 GB/512 GB SSD/Windows 10 Home/4 GB Graphics/NVIDIA GeForce GTX 1650 Ti) A715-41G-R7YZ Gaming Laptop  (15.6 inch, Charcoal Black, 2.15 kg)', 'ASUS ROG Strix G15 (2021) Ryzen 7 Octa Core 4800H - (8 GB/512 GB SSD/Windows 10 Home/4 GB Graphics/NVIDIA GeForce GTX 1650/144 Hz) G513IH-HN086T Gaming Laptop  (15.6 inch, Eclipse Gray, 2.10 Kg)', 'Lenovo IdeaPad 3 Core i3 10th Gen - (8 GB/1 TB HDD/Windows 10 Home) 15IML05 Thin and Light Laptop  (15.6 inch, Platinum Grey, 1.7 kg, With MS Office)', 'ASUS VivoBook 15 (2022) Core i3 10th Gen - (8 GB/512 GB SSD/Windows 11 Home) X515JA-EJ362WS Thin and Light Laptop  (15.6 inch, Transparent Silver, 1.80 kg, With MS Office)', 'ASUS VivoBook 15 (2022) Core i7 10th Gen - (16 GB/512 GB SSD/Windows 11 Home/Intel Integrated Iris Plus) X515JA-EJ701WS Gaming Laptop  (15.6 inch, Transparent Silver, 1.80 kg, With MS Office)']
user rating (float64, 28 distinct, 29.9% missing): ['4.3', '4.4', '4.5', '4.2', '4.1', '4.0', '4.6', '4.7', '3.8', '3.9']
Sales Package (object, 106 distinct): ['Laptop, Power Adaptor, User Guide, Warranty Documents', 'Laptop, battery, adapter, cables and user manuals', 'Laptop, Battery, Adapter, Cables and User Manuals', 'Laptop, Power Adaptor, User Guide', 'Laptop, Power Adapter, User Guide, Warranty Documents', 'Laptop, Power Adaptor, Battery, Warranty Documents, User Guide', 'Laptop, Battery, Power Adaptor, User Guide, Warranty Documents', '2 in 1 Laptop, Power Adaptor, User Guide, Warranty Documents', 'Laptop, Battery, Power Adapter, User Guide', 'Laptop, Manual, Adaptor']
Model Number (object, 712 distinct): ['Inspiron 3511', 'Vostro 3400', 'Inspiron 5410', '14ITL05', 'Inspiron 3515', '15IML05', 'G15-5511', '3511', 'G15-5515', 'E15']
Part Number (object, 923 distinct): ['82ND007VIN', '34W78PA#ACJ', 'D560804WIN9W', 'NU14A1INC43PN-MB', '130290120209', '130290120210', '130290120208', '90NB0SG3-M43670', '9S7-154322-418', 'D552154WIN9BE']
Model Name (object, 469 distinct, 27.9% missing): ['Inspiron 3511', 'Inspiron 5410', 'Inspiron 3515', 'Vivobook 15', 'Inspiron', 'Vivobook 14', 'G15-5515', 'Vivobook', 'Book Prime', 'IdeaPad 3']
Series (object, 257 distinct, 20.0% missing): ['Inspiron', 'Vostro', 'Pavilion', 'IdeaPad 3', 'G15', 'Modern 14', 'Aspire 5', '15s', 'Chromebook', 'Vivobook 15']
Color (object, 132 distinct): ['Black', 'Transparent Silver', 'Natural Silver', 'Platinum Silver', 'Silver', 'Platinum Grey', 'Slate Grey', 'Arctic Grey', 'Shadow Black', 'Graphite Grey']
Type (object, 9 distinct): ['Thin and Light Laptop', 'Gaming Laptop', 'Laptop', '2 in 1 Laptop', 'Notebook', 'Chromebook', '2 in 1 Gaming Laptop', 'Business Laptop', 'Creator Laptop']
Suitable For (object, 52 distinct): ['Processing & Multitasking', 'Gaming', 'Everyday Use', 'Gaming, Processing & Multitasking', 'Everyday Use, Processing & Multitasking', 'Processing & Multitasking, Gaming', 'Entertainment, Everyday Use, Performance, Processing & Multitasking, Travel & Business', 'Processing & Multitasking, Everyday Use', 'Entertainment, Performance, Processing & Multitasking', 'Entertainment, Everyday Use, Gaming, Performance, Processing & Multitasking, Travel & Business']
Power Supply (object, 115 distinct, 44.3% missing): ['65 W AC Adapter', '45 W AC Adapter', '65W AC Adapter', '45W AC Adapter', '150W AC Adapter', '65 W Smart AC power adapter', '180 W AC Adapter', '180W AC Adapter', '100W AC Adapter', '65W Adapter']
Battery Cell (object, 31 distinct, 30.9% missing): ['3 cell', '3', '3 Cell', '4 cell', '4 Cell', '2 cell', '4', '2 Cell', '6 Cell', '2']
MS Office Provided (object, 2 distinct): ['Yes', 'No']
Dedicated Graphic Memory Type (object, 5 distinct, 65.7% missing): ['GDDR6', 'GDDR5', 'DDR4', 'DDR5', 'GDDR3']
Dedicated Graphic Memory Capacity (object, 9 distinct, 68.5% missing): ['4 GB', '2 GB', '6 GB', '8 GB', '16 GB', '128 MB', '10 GB', '512 MB', '12 GB']
Processor Brand (object, 5 distinct): ['Intel', 'AMD', 'Apple', 'Qualcomm', 'MediaTek']
Processor Name (object, 30 distinct): ['Core i5', 'Core i3', 'Core i7', 'Ryzen 5 Hexa Core', 'Ryzen 7 Octa Core', 'Celeron Dual Core', 'Ryzen 9 Octa Core', 'Ryzen 3 Dual Core', 'Ryzen 5 Quad Core', 'Core i9']
Processor Generation (object, 9 distinct, 29.9% missing): ['11th Gen', '10th Gen', '12th Gen', '5th Gen', '4th Gen', '8th Gen', '9th Gen', '3rd Gen', '7th Gen']
SSD (object, 2 distinct): ['Yes', 'No']
SSD Capacity (object, 5 distinct, 11.0% missing): ['512 GB', '256 GB', '1 TB', '128 GB', '2 TB']
RAM (object, 4 distinct): ['8 GB', '16 GB', '4 GB', '32 GB']
RAM Type (object, 8 distinct): ['DDR4', 'LPDDR4X', 'DDR5', 'LPDDR4', 'LPDDR5', 'LPDDR3', 'Unified Memory', 'DDR3']
Processor Variant (object, 132 distinct, 12.5% missing): ['1135G7', '1115G4', '1005G1', '5500U', '11800H', '1165G7', 'N4020', '5800H', '1155G7', '3250U']
Clock Speed (object, 350 distinct, 12.5% missing): ['2.4 GHz with Turbo Boost Upto 4.2 GHz', '1.2 GHz upto max turbo frequency at 3.4 Ghz', '4.2', '2.1 GHz with Turbo Boost Upto 4.1 GHz', '4.1', '1.2 GHz with Turbo Boost Upto 3.4 GHz', 'Base Frequency 2.40 GHz, Max Turbo Boost Frequency Up to 4.20 GHz', '2.8 GHz with Turbo Boost Upto 4.7 GHz', '1.7 GHz upto max turbo frequency at 4.1 Ghz', '3.5']
Expandable Memory (object, 36 distinct, 61.1% missing): ['16', 'Upto 16 GB', '32', 'Upto 32 GB', '16 GB', '16GB', '12', 'Upto 12 GB', 'Upto 20 GB', '64']
Cache (object, 42 distinct, 22.7% missing): ['8', '8 MB', '4', '6', '6 MB', '16', '4 MB', '12', '24', '20']
Graphic Processor (object, 151 distinct, 2.0% missing): ['Intel Integrated UHD', 'Intel Integrated Iris Xe', 'NVIDIA GeForce GTX 1650', 'NVIDIA GeForce RTX 3050', 'AMD Radeon AMD', 'AMD Radeon', 'NVIDIA GeForce RTX 3060', 'Intel Integrated', 'Intel Integrated UHD Graphics', 'Intel Integrated Iris Xe Graphics']
Number of Cores (float64, 8 distinct, 26.4% missing): ['4.0', '2.0', '8.0', '6.0', '14.0', '12.0', '1.0', '10.0']
OS Architecture (object, 2 distinct, 16.1% missing): ['64 bit', '32 bit']
Operating System (object, 10 distinct): ['Windows 10 Home', 'Windows 11 Home', 'Windows 10', 'Chrome', 'Windows 10 Pro', 'Mac OS Big Sur', 'DOS', 'Windows 11 Pro', 'Mac OS Monterey', 'Ubuntu']
Supported Operating System (object, 34 distinct, 50.3% missing): ['Windows 11 Home', 'Windows 11', 'Windows 10 Home', 'Windows 10', 'Windows 10 Home 64-bit', 'Windows 11 Home 64-bit', 'Windows10 Home', 'Windows 11 Home 64 Plus Single Language', 'Windows 11 Home 64', 'Windows 10 Home 64, English']
Mic In (object, 2 distinct, 18.0% missing): ['Yes', 'No']
USB Port (object, 401 distinct, 10.6% missing): ['1 x USB 3.2 Gen 1 Type-A, 1 x USB 3.2 Gen 1 Type-C, 2 x USB 2.0 Type-A', '1x USB 3.2 Gen 1 Type-A, 1x USB 3.2 Gen 1 Type-C, 2x USB 2.0 Type-A', '1 x SuperSpeed USB Type-C 5Gbps signaling rate, 2 x SuperSpeed USB Type-A 5Gbps signaling rate', '2 x USB 3.2 Type A, 1 x USB 3.2 Type C', '1 x USB 2.0, 2 x USB 3.2 Gen 1', '2 x USB 3.2 Gen 1, 1 x USB2.0', '1 x SuperSpeed USB Type C (5 Gbps Signaling Rate), 2 x SuperSpeed USB Type A (5 Gbps Signaling Rate)', '1 x USB 3.2 (1st Gen) Type A, 1 x USB 3.2 (1st Gen) Type C, 2 x USB 2.0 Type A', '2 x USB 3.2 Gen1 ,1 x USB 2.0', '1 x USB 3.2 Gen 1 Type-A,1 x USB 3.2 Gen 1 Type-C, 2 x USB 2.0 Type-A']
HDMI Port (object, 65 distinct, 18.3% missing): ['1 x HDMI 1.4', '1 x HDMI Port', '1x HDMI 1.4 port', '1 x HDMI Port (v1.4b)', '1 x HDMI Port (v1.4)', '1x HDMI 1.4', '1 x HDMI 2.0b', '1x HDMI 2.0b', '1 x HDMI 1.4b', '1 x HDMI']
Touchscreen (object, 2 distinct): ['No', 'Yes']
Screen Size (object, 38 distinct): ['39.62 cm (15.6 inch)', '35.56 cm (14 inch)', '39.62 cm (15.6 Inch)', '35.56 cm (14 Inch)', '33.78 cm (13.3 inch)', '39.62 cm (15.6 inches)', '33.78 cm (13.3 Inch)', '43.94 cm (17.3 inch)', '35.56 cm (14 inches)', '40.89 cm (16.1 inch)']
Screen Resolution (object, 72 distinct): ['1920 x 1080 Pixel', '1920 x 1080 Pixels', '1920 x 1080 pixel', '1366 x 768 Pixel', '1920 x 1080$$Pixels pixel', '1366 x 768 Pixels', '2560 x 1440 Pixel', '2560 x 1600 Pixel', '1920x1080 pixel', '1366 x 768 pixel']
Screen Type (object, 518 distinct, 10.3% missing): ['Full HD WVA AG Narrow Border', 'FHD WVA AG Narrow Border', 'Full HD Anti Glare', 'Full HD IPS Display ( 300nits peak brightness, 100% sRGB, NTSC 72%)', 'Full HD (IPS-level Panel, 250nits, 45% NTSC color gamut, non-OLED, Anti-glare display)', 'Full HD Anti Glare Display', 'Full HD WVA AG Narrow Border Display', 'FHD, OLED (550nits peak brightness, 100% DCI-P3 color gamut)', 'Full HD WVA AG 250 nits 120Hz Narrow Border', 'Full HD LED Backlit, Anti-Glare Display (250 nits Brightness)']
Speakers (object, 76 distinct, 13.4% missing): ['Built-in Dual Speakers', 'Built-in Speakers', 'Yes', 'Built-in Speaker', 'Built-in speaker', 'Built-in Stereo Speakers', 'Dual speakers', 'Built-in Dual Stereo Speakers', 'Dual Speakers', 'Built-In Dual Stereo Speakers']
Internal Mic (object, 69 distinct, 14.3% missing): ['Yes', 'Built-in Microphone', 'Built-in array microphone', 'Built-in microphone', 'Built-in Microphones', 'Built-in Array Microphone', 'Built-in Dual Microphone', 'Built-in dual array digital microphones', 'Integrated Dual Array Digital Microphone', 'Built-In Dual Array Microphone']
Sound Properties (object, 203 distinct, 29.1% missing): ['Waves Maxx Audio Pro', 'Audio by B&O', 'Nahimic 3D Audio', '2 x 1.5W Stereo Speakers with Dolby Audio', 'Audio by Bang & Olufsen', 'SonicMaster', 'SonicMaster Audio by ICEpower', 'Audio by ICEpower', '2 x 1.5 W Stereo Speakers with Dolby Audio', 'DTS Audio Processing']
Wireless LAN (object, 170 distinct, 11.8% missing): ['802.11ac', 'Wi-Fi 6(802.11ax)', 'Intel Wi-Fi 6 2x2 (Gig+)', '11ac, 2x2', 'Wi-Fi 5(802.11ac)', 'Realtek RTL8822CE 802.11a/b/g/n/ac (2x2) Wi-Fi', 'IEEE 802.11ac', '802.11ax', 'Intel Wi-Fi 6 AX201(2*2 ax)', 'Realtek Wi-Fi 6 (2x2)']
Bluetooth (object, 26 distinct, 0.4% missing): ['v5.0', 'v5.1', 'v5.2', 'v4.1', 'YES', 'v4.2', 'yes', 'Yes', '4.2', '4.1']
Dimensions (object, 380 distinct, 12.3% missing): ['360.2 x 234.9 x 19.9 mm', '358 x 235 x 18.9 mm', '189 X 358 X 23.5 mm', '362.2 x 253.4 x 19.9 mm', '240 x 329 x 20', '358 x 242 x 19.9 mm', '324 x 225 x 17.9 mm', '358 x 242 x 17.9 mm', '357 x 272 x 26.9 mm', '360 x 235 x 19.9 mm']
Weight (object, 207 distinct, 11.3% missing): ['1.80 kg', '1.8 kg', '1.7 kg', '1.85 kg', '1.40 kg', '1.5 kg', '2.25 kg', '1.6 kg', '2.30 kg', '1.8 Kgs']
Disk Drive (object, 3 distinct): ['Not Available', 'CD/DVD writer', 'CD/DVD reader']
Finger Print Sensor (object, 2 distinct, 27.4% missing): ['No', 'Yes']
Keyboard (object, 189 distinct, 10.3% missing): ['Backlit Chiclet Keyboard', 'English International Non Backlit Keyboard', 'Backlit Keyboard', 'Chiclet Keyboard', 'Chiclet Keyboard with Num-key', 'English Backlit Keyboard', 'Standard English Backlit Keyboard', 'English International Backlit Keyboard', 'Backlit, English', 'Backlit Chiclet Keyboard with Num-key']
Backlit Keyboard (object, 2 distinct, 17.1% missing): ['Yes', 'No']
Additional Features (object, 296 distinct, 41.4% missing): ['Li-ion Battery', '42WHrs Li-ion', '37 WHrs Li-ion Battery', '37WHrs Li-ion', '37 WHr Li-ion Battery, BIOS Booting User Password Protection', '41 Wh Li-ion, Support battery fast charge', '59 Wh 4-cell Li-ion?battery', '90WHrs Li-ion', '41 Wh Li-ion Battery, Support battery fast charge', 'Google Assistant Voice-recognition Support, TYPE-C, 45W AC Adapter, Output: 15V DC, 3A, 45W, Input: 100~240V AC 50/60Hz Universal']
Warranty Summary (object, 81 distinct): ['1 Year Onsite Warranty', '1 Year International Travelers Warranty (ITW)', '1 Year Warranty + 1 Year Premium Care + 1 Year ADP', '2 Years Warranty', '1 Year Warranty', '1 Year Premium Support', '1 Year onsite warranty', '1 YEAR', '2 Year Warranty Term', '1 Year Limited Hardware Warranty, In Home Service After Remote Diagnosis - Retail']
Warranty Service Type (object, 25 distinct): ['Onsite', 'Carry-In Warranty', 'ONSITE', 'Carry-in', 'onsite', 'Carry in', 'on site', 'Carry-In', 'Carry In', 'ON SIDE']
Covered in Warranty (object, 36 distinct): ['Manufacturing Defects', 'Part Failure', '(1) Manufacturer�s warranty against faulty workmanship or defective parts; (2) ADP-Single repair once in year against all Liquid spills, unintentional bump and drops, electric surge, cracks on screen; (3) Premium Care- 24x7 on call support, Software support in case of HDD failure', 'Manufacturer�s warranty against faulty workmanship or defective parts', 'MANUFACTURING DEFECT', 'MANUFACTURING DAMAGE', 'Manufacturing defects', 'Manufacturer Defect', '(1) Manufacturer�s warranty against faulty workmanship or defective parts; (2) ADP-Single repair once in year against all Liquid spills, unintentional bump and drops, electric surge, cracks on screen; (3) Gamer centric support 24x7 on call support, Software support in case of HDD failure', 'Manufacturer defect']
Not Covered in Warranty (object, 42 distinct, 0.3% missing): ['Physical Damage', 'Accidental Damage', '(1) Any kind of physical damage including electrical surge; (2) No software coverage in warranty; (3) Premium Care & ADP- Not covered in case of Theft, fire, rain, flood and part alteration', 'Physical Damage, Burn, Liquid Spill', '(1) Any kind of physical damage including electrical surge; (2) No software coverage in warranty', 'PHYSICAL DAMAGE', 'Physical damage', 'Any Type of Damage', 'Damage, Burn', 'Physical Damage|| Burn|| Liquid Spill']
Domestic Warranty (object, 6 distinct, 17.3% missing): ['1 Year', '2 Year', '3 Year', '12 Months', '36 Months', '24 Months']
Ethernet (object, 30 distinct, 68.5% missing): ['Available', 'Yes', 'Integrated 10/100/1000 GbE LAN', '10/100/1000 Mbps', '100/1000M', 'Not Available', 'Gigabit Ethernet, Wake-on-LAN Ready', 'Killer Ethernet E2600', 'Gigabit Ethernet', '10/100/1000 GbE LAN']
Web Camera (object, 140 distinct, 14.3% missing): ['HD', 'HD Webcam', '720p HD Webcam', '720p HD camera', 'HP True Vision 720p HD camera', 'VGA camera', 'HP Wide Vision 720p HD camera', 'Yes', '720p with Privacy Shutter', '720P HD camera']
Pointer Device (object, 46 distinct, 41.4% missing): ['Touchpad', 'HP Imagepad with multi-touch gesture support', 'Touchpad with Multi-touch Gesture Support', 'Multi-touch Touchpad', 'MultiGesture Touchpad', 'HP Imagepad with Multi-touch Gesture Support', 'Touchpad and Active Pen', 'Force Touch Trackpad', 'Touchpad with Multi-Touch Gesture Support', 'Smooth surface muilt-touch Trackpad with Gesture support']
Included Software (object, 171 distinct, 38.6% missing): ['Microsoft Office Home and Student 2019', 'Office Home and Student 2021', 'Microsoft Office Home & Student 2019', 'Microsoft Office Home and Student 2021', 'Office Home and Student 2019 included', 'Acer Care Center, Quick Access, Acer Product Registration', 'Office Home and Student 2019', 'Microsoft Office Home & Student 2021', 'Microsoft Office 2019 Home & Student', 'Built-in Apps: iMovie, Siri, GarageBand, Pages, Numbers, Photos, Keynote, Safari, Mail, FaceTime, Messages, Maps, Stocks, Home, Voice Memos, Notes, Calendar, Contacts, Reminders, Photo Booth, Preview, Books, App Store, Time Machine, TV, Music, Podcasts, Find My, QuickTime Player']
Battery Backup (object, 152 distinct, 38.2% missing): ['Upto 10 Hours', 'Upto 7 hours', 'Upto 6 hours', 'Upto 6 Hrs.', 'Upto 12 hours', 'Upto 8 hours', 'UPTO 6 HRS', 'Upto 10 hours', '3 Hours', 'Upto 6 Hours']
Chipset (object, 29 distinct, 86.0% missing): ['Intel SoC Platform', 'AMD SoC Platform', 'Intel SoC', 'AMD Integrated SoC', 'AMD SoC', 'Intel HM470', 'Intel i5-11320H', 'Intel Integrated SoC', 'Intel HM570', 'Mobile Intel HM470 Express Chipsets']
Memory Slots (object, 17 distinct, 69.5% missing): ['2', '2 Slots', '1', '1 Slot', 'DDR4 SO-DIMM * 2', 'Two DDR4 SO-DIMM slots, dual-channel capable', '1x SO-DIMM', '0', 'Memory soldered to systemboard, no slots, dual-channel', 'One memory soldered to systemboard, one DDR4 SO-DIMM slot, dual-channel capable']
RAM Frequency (object, 19 distinct, 60.1% missing): ['3200 MHz', '2666 MHz', '2933 MHz', '2400 MHz', '4266 MHz', '3733 MHz', '4800 Mhz', '4800 MHz', '2400 Mhz', '4267 MHz']
RJ45 (object, 2 distinct, 57.5% missing): ['Yes', 'No']
Sound Chip (object, 33 distinct, 88.7% missing): ['High Definition (HD) Audio', 'High Definition (HD) Audio, Realtek ALC3287 codec', 'Realtek ALC233', 'Realtek ALC 233', 'Dolby Atmos�', 'Dolby Atmos', 'Realtek ALC 3287 codec, High Definition (HD) Audio', 'Sonic Master', 'Realtek HD Audio', 'Audio by ICEpower�']
Brightness (object, 4 distinct, 98.8% missing): ['300 Nits', '250 Nits', '400 Nits', '220 Nits']
Laptop Bag (object, 2 distinct, 80.8% missing): ['No', 'Yes']
Other Accessories (object, 18 distinct, 96.0% missing): ['Power Adaptor', 'Sleeve', 'Sleeve, Stand, Stylus', 'Active Pen', 'Dell Active Pen', 'Sleeve Bag, Pen', 'Sleeve, USB-C to Audio Jack Adapter', 'GV301 Sleeve Bag, Pen', 'NO', 'Mouse(ROG P512), Mouse Pad(ROG)']
International Warranty (object, 6 distinct, 85.7% missing): ['1 Year', '2 Year', '12 Months', '0 Months', '36 Months', '1']
Wireless WAN (object, 30 distinct, 93.7% missing): ['Yes', 'Wi-Fi 5 (802.11ac) 2*2', 'YES', 'Wi-Fi 6(802.11ax)+Bluetooth 5.0 (Dual band) 2*2', 'Wi-Fi 6(802.11ax)', 'Wi-Fi 6(802.11ax)+Bluetooth 5.1 (Dual band) 2*2', 'Intel Wi-Fi 6 AX201(2*2 ax)', 'Wi-Fi 5(802.11ac)', 'Wi-Fi 5 (802.11ac) 1*1', 'Intel Wireless 9462AC Card (80 2.11ac + Bluetooth 5.0, Dual B and 2.45 GHz, MU-MIMO/80Mhz, 1 x1)']
Recovery Options (object, 8 distinct, 96.7% missing): ['Yes', 'One Touch Recovery', 'One Button', 'Na', 'ONE BUTTON', 'HARD DRIVE PARTITION, CLOUD DOWNLOAD', 'INBUILT', 'yes']
RPM (float64, 6 distinct, 90.5% missing): ['5400.0', '7200.0', '4.0', '2666.0', '3200.0', '8.0']
Hardware Interface (object, 47 distinct, 45.1% missing): ['PCIe NVMe', 'M.2 SATA', 'PCIe NVMe M.2', 'NVMe PCIe', 'SATA', 'NVMe� PCIe� 3.0 SSD', 'M.2 NVMe PCIe 4.0', 'PCIe NVMe TLC M.2', 'M.2 Gen 4', 'PCIE 3.0']
Face Recognition (object, 2 distinct, 96.5% missing): ['No', 'Yes']
System Architecture (object, 11 distinct, 42.5% missing): ['64', '64 bit', '64 Bit', '64 bits', 'X64', 'x64', '64Bits', '64Bit', '64 BIT', '64-Bit']
Refresh Rate (object, 11 distinct, 84.0% missing): ['144 Hz', '60 Hz', '120 Hz', '165 Hz', '300 Hz', '240 Hz', '90 Hz', '360 Hz', '144 HZ', '60 HZ']
Antivirus (object, 15 distinct, 86.7% missing): ['McAfee', '1 Year McAfee', 'McAfee Multi Device Security 15 Months Subscription', 'McAfee LiveSafe', 'Mcafee Antivirus - 1 Year', 'YES', 'McAfee (15 months Subscription)', 'Mcafee AntiVirus - 1 Year', 'McAfee(R) Multi Device Securit y 15 month subscription', '1 Year Antivirues']
Multi Card Slot (object, 72 distinct, 48.6% missing): ['1 x Micro SD Card Reader', '1 x Micro SD card reader', 'Micro SD Card Reader', '1 x card reader', '3-in-1 Card Reader (SD, SDHC, SDXC)', '1 x SD Media Card Reader (SD, SDHC, SDXC)', 'SD Media Card Reader (SD, SDHC, SDXC)', '4-in-1 Card Reader', '1x Micro SD card reader', '1x SD Media Card Reader']
Lock Port (object, 17 distinct, 77.2% missing): ['Kensington Lock Port', 'Kensington Lock Slot', 'Security Lock', 'Kensington lock slot', 'Kensington Security Slot', 'Kensington Lock', 'Function key lock', 'Energy Star Certified, EPEAT Silver Registered', 'Kensington Security Lock', 'Yes']
NFC Support (object, 1 distinct, 95.8% missing): ['No']
HDD Capacity (object, 2 distinct, 81.5% missing): ['1 TB', '512 GB']
Stylus Included (object, 2 distinct, 94.8% missing): ['No', 'Yes']
TGP (object, 12 distinct, 97.1% missing): ['35 W', '60 W', '50 W', '140 W', '75 W', '240 W', '10 W', '95 W', '85 W', '65 W']
VGA Port (object, 2 distinct, 90.8% missing): ['No', 'Yes']
Color Gamut (object, 3 distinct, 99.6% missing): ['100% DCI-P3 color gamut, PANTONE Validated', '75%', '75']
Security Chip (object, 19 distinct, 85.6% missing): ['Firmware TPM 2.0', 'TPM 2.0', 'TPM', 'Trusted Platform Module (Firmware TPM)', 'Energy Star Certified, EPEAT Silver Registered', 'Titan C Security Chip', 'BIOS Administrator Password and User Password Protection', 'Hardware TPM 2.0', 'Kensington Security Slot', 'Fingerprint Sensor Integrated with Power Key, Trusted Platform Module (TPM)']
Dock Port (object, 1 distinct, 96.6% missing): ['No']
Firewire Port (object, 1 distinct, 96.5% missing): ['No']
RJ11 (object, 1 distinct, 95.8% missing): ['No']
EMMC Storage Capacity (object, 4 distinct, 97.6% missing): ['64 GB', '128 GB', '4 GB', '32 GB']
'''

def load_df(dir_path: str) -> DataFrame:
    df_path = join(dir_path, "complete laptop data0.csv")
    df = read_csv(df_path, encoding="utf-8", encoding_errors="replace")
    df['Price'] = df['Price'].apply(clean_rupi)
    return df


def clean_rupi(price: str) -> float:
    # Rupi looks like: ?2,34,990
    assert price.startswith("?")
    price = price.strip("?").replace(",", "")
    return float(price)

CONTEXT = ""
COLS_TO_DROP = [
    # Index
    "Unnamed: 0",
    # Non-Image URL
    "link",
    # Constant / Highly missing
    "Cloud Storage", "Optane Memory", "Read/Write Speed", "S-video", "Inbuilt 4G LTE",
    ]
TARGET = CuratedTarget(raw_name="Price", task_type=SupervisedTask.REGRESSION)
LOADING_FUNC = load_df