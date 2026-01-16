import os
from os.path import exists, join
from typing import Optional

from pandas import DataFrame

from tabstar2.utils.datasets import load_csv
from tabstar_paper.datasets.curation_objects import CuratedTarget, CuratedFeature
from tabstar_paper.datasets.objects import SupervisedTask, FeatureType


'''
Dataset Name: laurinbrechter/online-boat-listings/
====
Examples: 1850
====
URL: https://www.kaggle.com/laurinbrechter/online-boat-listings
====
Description: <COPY PASTE HERE FROM KAGGLE URL>
====
Target Variable: price (float64, 798 distinct): ['65000.0', '45000.0', '110000.0', '75000.0', '55000.0', '79000.0', '59000.0', '120000.0', '125000.0', '115000.0']
====
Features:

name (object, 1752 distinct): ['2015 Beneteau Oceanis 452015 Beneteau Oceanis 45', '2022 Jeanneau Sun Odyssey 4902022 Jeanneau Sun Odyssey 490', '1988 LE GUEN HEMIDY SA LEVRIER DES MERS 14M1988 LE GUEN HEMIDY SA LEVRIER DES MERS 14M', '2021 Dufour 5302021 Dufour 530', '2017 Beneteau Oceanis 482017 Beneteau Oceanis 48', '2022 Bavaria Cruiser 342022 Bavaria Cruiser 34', '2005 Bavaria Cruiser 462005 Bavaria Cruiser 46', '2016 Dufour 460 Grand Large2016 Dufour 460 Grand Large', '1968 Boudignon KETCH CLASSIQUE FLAMANT 111968 Boudignon KETCH CLASSIQUE FLAMANT 11', '2018 Robertson & Caine Leopard 402018 Robertson & Caine Leopard 40']
location (object, 702 distinct): ['cannes, alpes-maritimes ( 06 )', 'in verkoophaven, niederlande', 'breege, deutschland', 'lisboa, portugal', 'levington, suffolk', 'olbia, italien', 'norddeutschland, schleswig-holstein', 'türkei', 'mugla, türkei', 'contact de valk hindeloopen, niederlande']
offerer (object, 135 distinct): ['Privater Verkäufer', 'De Valk Group B.V', 'XBOAT', 'Clarke & Carter Interyacht Ltd.', 'White Whale Yachtbrokers', 'Michael Schmidt & Partner GmbH', 'Sea Way Lda', 'Bernard Gallay Yacht Brokerage', 'T.Y.Broker Srls', 'Gino Group']
description (object, 1802 distinct): ['---Show more', '-Show more', 'Make: Beneteau Model: Bénéteau Océanis 45 Year: 2015Mehr anzeigen', '1968 FLAMINGO 11 CLASSIC KETCH BOUDIGNON,\nXBOAT ref 5383. Classic boat classified as a Boat of Heritage Interest (BIP), in good condition and maintenance. The original wooden mast was replaced in 2013 / 2014 by an aluminum mast. Standing rigging replaced in 2013 / 2014. Running rigging replaced in 2016. Mobile solar panel.\n\n1 DSC VHF + 1 portable VHF + AIS More information and photos on the site xboat.frMehr anzeigen', 'REMOTE PURCHASE AVAILABLE\xa0-Find the boat\xa0-Agree on the deal-Arrange an international surveyor (either for pre-survey or full survey)-Conference call viewings for more-Remote paperwork\xa0-Close the deal-Manage the transfer or transport to your desired destination and start your new journeyMehr anzeigen', '-\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b \u200b \u200b \u200bShow more', 'Make: Beneteau Model: Bénéteau Océanis 48 Year: 2017Mehr anzeigen', 'Make: Robertson & Caine (SA) Model: Leopard 40 Year: 2018Show more', '2006 NAUTITECH NAUTITECH 40,\nXBOAT ref 2977. Boat easy to maneuver, great habitability.\n\nThe Nautitech 40 is a comfortable and safe boat. Everything is done to make you feel in a cozy and reassuring cocoon, both on deck and in the cabin. the Nautitech 40 offers great performance at sea.\n\nThis Nautitech 40 has been well maintained in the Seychelles since (2011), she was previously in the Mediterranean on charter with a pro skipper, around 8 weeks a year.\n\nEngines (2013) and (2016), standing rigging 2011 for the shrouds and (2013) stays, windlass (2012), trampoline (2015), furler (2013), teak exterior (2013), gas cooker (2015), mattress covers (2014), interior floors (2014), led lighting, sails 2018, Garmin electronics (2018) last refit July (2019).\n\nvisit video on request More information and photos on the site xboat.frMehr anzeigen', 'Make: Robertson & Caine (SA) Model: Leopard 40 Year: 2017Show more']
year (int64, 97 distinct): ['2022', '2008', '2005', '2007', '2006', '1980', '2017', '2018', '1983', '1984']
manufacturer (object, 606 distinct): ['Jeanneau', 'Bavaria', 'Beneteau', 'Dufour', 'Custom', 'Dehler', 'Hanse', 'Hallberg-Rassy', 'Lagoon', 'Grand Soleil']
model (object, 1322 distinct, 0.4% missing): ['34', '38', '40', '32', '44', '50', '36', '37', '42', 'Cruiser 46']
category (object, 25 distinct): ['Segelyachten', 'Slup Boote', 'Daysailers', 'Racer/ Cruiser Segelboote', 'Sonstige (Segelboote)', 'Klassische Segelboote', 'Katamarane', 'Motorsegler (Segel)', 'Racer Segelboote', 'Katamarane und Trimarane']
length (float64, 688 distinct): ['9.0', '8.0', '12.5', '13.0', '11.3', '9.5', '10.0', '12.0', '9.99', '11.0']
fuel_type (object, 4 distinct, 6.2% missing): ['diesel', 'petrol', 'electrical', 'other']
hull_material (object, 8 distinct): ['GFK / fiberglas / polyester', 'other', 'composite', 'wood', 'steel', 'Aluminium', 'PVC', 'ferrocement']
hull_shape (object, 9 distinct, 78.3% missing): ['other', 'Monohull ', 'Verdränger', 'Katamaran', 'Halbgleiter', 'Trimaran', 'Tunnel', 'Deep Vee', 'Gleiter']
country (object, 36 distinct): ['afghanistan', 'germany', 'netherlands', 'italy', 'spain', 'turkey', 'france', 'croatia', 'portugal', 'greece']
boat_image (object, 1601 distinct, 13.5% missing): ['8222682.jpg', '6751543.jpg', '469955.jpg', '8404683.jpg', '7937781.jpg', '6081675.jpg', '7936739.jpg', '8293536.jpg', '469900.jpg', '470478.jpg']
'''

BOAT_IMAGE = "boat_image"


def load_df(dir_path: str) -> DataFrame:
    df = load_csv(dir_path, "data_clean.csv")
    # df_raw = load_csv(dir_path, "data_raw.csv")
    df[BOAT_IMAGE] = df['id'].apply(lambda id: parse_img(id, dir_path))
    return df


def parse_img(img: str, data_dir: str) -> Optional[str]:
    path = f"{img}.jpg"
    if not exists(join(data_dir, IMAGE_FOLDER, path)):
        return None
    return path




CONTEXT = ""
TARGET = CuratedTarget(raw_name='price', task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = ["Unnamed  0", "id", 'Unnamed  0']
TEXT_FEATURES = [CuratedFeature(raw_name=f, feat_type=FeatureType.TEXT) for f in ['name', 'location', 'offerer', 'description', 'manufacturer', 'model']]
FEATURES = [CuratedFeature(raw_name=BOAT_IMAGE, feat_type=FeatureType.IMAGE)] + TEXT_FEATURES
IMAGE_FOLDER = "images/images"
LOADING_FUNC = load_df
