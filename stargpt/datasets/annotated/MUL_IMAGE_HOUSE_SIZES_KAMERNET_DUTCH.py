import os
from os.path import exists, join
from typing import List

from pandas import DataFrame, read_csv

from tabstar2.utils.datasets import load_csv
from tabstar_paper.datasets.curation_objects import CuratedTarget, CuratedFeature
from tabstar_paper.datasets.objects import SupervisedTask, FeatureType


'''
Dataset Name: amancapy/all-kamernet-rent-listings-w-photos-netherlands/
====
Examples: 1659
====
URL: https://www.kaggle.com/amancapy/all-kamernet-rent-listings-w-photos-netherlands
====
Description: 
All Kamernet Rent Listings w. Photos (Netherlands)
All listing pages on a given day scraped along with listing photos

About Dataset
All visible details on listing pages were scraped and cleaned up for tabulation, except the age of the landlord's account and the number of times they have rented a property out before. These will be added in future versions. Images were saved at thumbnail resolution (usually w:256px) so as not to make expensive calls to the website in such great quantities.

The set of listings recognized in a given scrape is that of those online between the earliest time_of_checking and roughly 30 minutes ahead from then, at the latter of which the scraper finishes scraping search pages and starts focusing solely on individual listing pages. A drawback here is that if a new listing goes online after the first search page has already been scraped, it will not be recognized. Since in all scraping takes roughly 5-6 hours, we can expect upwards of a dozen new listings missed on an active day.

Please note that the tabular data visualizer below is inaccurate since it trips up on commas within string columns. Any competent csv library will parse it correctly.

I will rescrape periodically but surely not at regular intervals. I don't want to get rate-limited by the website by scraping conspicuously too often.
====
Target Variable: furnish_type (object, 3 distinct): ['furnished', 'unfurnished', 'uncarpeted']
====
Features:

place_type (object, 3 distinct, 0.7% missing): ['room', 'apartment', 'studio']
price (int64, 385 distinct): ['700', '450', '750', '650', '850', '600', '500', '1000', '550', '950']
deposit (float64, 322 distinct, 25.0% missing): ['1000.0', '500.0', '600.0', '1500.0', '700.0', '300.0', '800.0', '650.0', '750.0', '200.0']
additional_costs (float64, 78 distinct, 83.5% missing): ['100.0', '25.0', '150.0', '200.0', '50.0', '15.0', '75.0', '175.0', '80.0', '120.0']
additionals (object, 231 distinct): ['Unknown Value', 'Servicecosts', 'Utilities', 'Internet', 'G/W/E', 'gas / water / electricity / internet / inventory / furnishings / waste tax', 'Wifi', '-', 'water, heating and electricity', 'service costs']
size (int64, 123 distinct): ['12', '20', '16', '15', '10', '14', '18', '25', '13', '17']
location (object, 1363 distinct): ['enschede, calslaan', 'enschede, campuslaan', 'enschede, deurningerstraat', 'enschede, witbreuksweg', 'wageningen, hoogstraat', 'delft, jacoba-van-beierenlaan', 'delft, e-du-perronlaan', 'enschede, lipperkerkstraat', 'delft, oude-delft', 'heerlen, coriovallumstraat']
posted_ago (object, 47 distinct): ['3 days', '2 days', '1 days', '8 days', '10 days', '4 days', '7 days', '6 days', '5 days', '9 days']
time_of_checking (object, 1659 distinct): ['2025-07-11 02:19:18', '2025-07-11 06:26:54', '2025-07-11 04:42:42', '2025-07-11 02:52:48', '2025-07-11 04:09:20', '2025-07-11 04:02:18', '2025-07-11 05:42:37', '2025-07-11 03:19:24', '2025-07-11 01:36:03', '2025-07-11 02:59:50']
from_date (object, 123 distinct): ['1 aug 2025', '1 jul 2025', '1 sep 2025', '15 jul 2025', '7 jul 2025', '10 jul 2025', '14 jul 2025', '8 jul 2025', '5 jul 2025', '9 jul 2025']
to_date (object, 184 distinct): ['inf', '31 aug 2025', '31 jul 2026', '31 jan 2026', '31 dec 2025', '1 sep 2025', '1 aug 2026', '30 jun 2026', '30 sep 2025', '1 feb 2026']
n_tenants (object, 6 distinct): ['1', '2', 'Not important', '5', '4', '3']
age_range (object, 257 distinct): ['16 - 99', 'Not important', '16 - 30', '18 - 35', '18 - 25', '18 - 30', '18 - 99', '18 - 28', '16 - 25', '16 - 26']
gender (object, 4 distinct): ['Not important', 'Female', 'Male', 'Everyone welcome']
occupation (object, 13 distinct): ['Student , Working student', 'Student', 'Student , Working student , Working', 'Everyone welcome', 'Student , Working student , Working , Looking for a job', 'Working student , Working', 'Working', 'Working student , Working , Looking for a job', 'Working , Looking for a job', 'Student , Working student , Looking for a job']
languages (object, 60 distinct): ['Everyone welcome', 'Dutch', 'Dutch , English', 'English', 'Dutch , English , German', 'Dutch , English , Spanish', 'Dutch , English , French , German', 'Dutch , English , Spanish , German', 'Dutch , English , French', 'Dutch , English , Spanish , French , German']
living_room (object, 3 distinct, 5.8% missing): ['shared', 'private', 'no']
kitchen (object, 3 distinct, 4.3% missing): ['shared', 'private', 'no']
bathroom (object, 3 distinct, 4.4% missing): ['shared', 'private', 'no']
toilet (object, 2 distinct, 4.2% missing): ['shared', 'private']
internet (object, 2 distinct, 8.5% missing): ['1', '0']
energy_label (object, 8 distinct, 29.1% missing): ['unknown', 'a', 'c', 'b', 'd', 'e', 'g', 'f']
n_roommates (float64, 9 distinct, 10.2% missing): ['0.0', '1.0', '2.0', '4.0', '8.0', '3.0', '5.0', '6.0', '7.0']
roommates_gender (object, 3 distinct, 27.1% missing): ['Mixed', 'Male', 'Female']
pets_allowed (bool, 2 distinct): ['0', '1']
price_includes_utilities (bool, 2 distinct): ['1', '0']
description (object, 1647 distinct): ["This nice furnished room at the Palmstraat 61 will become available on the 1st of August. The room can be rented for one year (fixed period of 1 year) or with a campus contract, which means you can rent the room as long as you're studying. [newline]- Rent is €650,-/month, including all costs (G/W/E, internet, etc.) [newline]- Deposit is €800,-", 'You live here in a quiet and safe part of Gronau, close to the Dutch border. The train station is 10 minutes away by bike and you have a direct train connection to Enschede Central Station. Alternatively, you can also take the bus (bus stops across the street) to the Gronauer train station. You can also cycle to the border in 10 minutes and use a bus connection there. You have all the important shopping facilities in the immediate vicinity. There also is a gym nearby. The flat is furnished with a fitted kitchen and has a large living room. The rooms are rented unfurnished. There are parking spaces for cars, but also a shelter for bicycles. We also have an internet connection which is included in the price. The kitchen (with dishwasher) is equipted with all the important things and there is a table with room for four persons. The bathroom has a washing mashine, a bath tub, a shower and two sinks. You can rent the rooms immedeatly. They are 16-28 m2 large.', "🏡 Furnished room for rent in Maastricht – peaceful living in a friendly student house [newline] [newline]!! At this moment it's rented by four girl students, and we're looking for someone who is comfortable with that [newline] [newline]Are you looking for a quiet and comfortable place to study and feel at home? In de Penatenhof, a small and peaceful student house in Maastricht, a furnished room will soon become available. Perfect for a student or working student who values cleanliness, comfort, and a great location. [newline] [newline]🛏️ Your room – move-in ready and well-equipped [newline]The room is fully furnished, so you can move in hassle-free. It includes: [newline] [newline]A three-door wardrobe with plenty of storage space [newline] [newline]A comfortable single bed + mattress [newline] [newline]A desk and chair for focused study time [newline] [newline]A private sink with mirror – convenient and hygienic [newline] [newline]Neatly finished, with lots of natural light [newline] [newline]📌 Minimum rental period: 12 months [newline] [newline]🏠 The house – calm, social, and respectful [newline]You’ll be sharing with a small group of fellow (master’s and working) students, all focused on their studies or work. The atmosphere is quiet, but there's always room for a friendly chat or shared meal. [newline] [newline]What to expect: [newline] [newline]Clear agreements [newline] [newline]Respectful environment [newline] [newline]No pets – ideal for those with allergies [newline] [newline]🍽️ Shared facilities [newline]1 bathroom with shower [newline] [newline]2 separate toilets – no waiting! [newline] [newline]Fully equipped kitchen with: [newline] [newline]Fridge [newline] [newline]Dishwasher [newline] [newline]Oven [newline] [newline]Microwave [newline] [newline]Plenty of cupboard space [newline] [newline]📍 Great location in Maastricht-West [newline]In a quiet, safe residential area [newline] [newline]2 min by bike to Daalhof shopping center (Lidl, bakery, pharmacy, drugstore) [newline] [newline]5 min by bike to Brusselse Poort (50+ stores) [newline] [newline]10 min by bike to Maastricht city center, university or university of applied sciences [newline] [newline]Bus stop nearby and free parking on the street. [newline] [newline]✅ Essentials [newline]Including utilities: gas, water, electricity, internet [newline] [newline]Deposit: €1,000 [newline] [newline]Only suitable for students or working students [newline] [newline]No registration fees or hidden costs [newline] [newline]🎓 Looking for a calm, reliable place where you can truly focus and feel at home? This might be your room! [newline]📩 Feel free to send us a message via Kamernet to schedule a viewing – you’re very welcome!", "About the accommodation [newline] [newline]This fully furnished 3-bedroom apartment offers a comfortable and well-equipped living environment, ideal for international students. Each bedroom features light wood-effect flooring, study desks, wardrobes, and a private TV for personal use. One of the rooms provides direct access to the balcony, offering a quiet outdoor space. The rooms are tastefully decorated with mirrors, shelving units, and natural light through large windows. Cozy seating, decorative fireplaces, and practical storage options are included throughout the rooms. [newline] [newline]The shared areas include a dining space with a wooden table and white chairs, positioned near a large window overlooking greenery. The galley kitchen is fully equipped with white cabinets, a gas stove, a built-in oven, a sink, and ample countertop space. A separate toilet room and a main bathroom with a bathtub, glass shower screen, sink, mirror, and washing machine are also available. With essential furnishings and shared facilities, this apartment provides a functional and pleasant home for student life. [newline] [newline]This is a full serviced accommodation, meaning it comes with furniture, appliances, utilities, and additional services included to ensure a hassle-free stay. Learn more here: What is a full serviced accommodation? [newline] [newline]Location [newline] [newline]Nestled in the Noordpolderbuurt area, this apartment is an ideal choice for THUAS students, with the university within biking distance. For students attending other universities, worry not, as public transportation options are available. [newline] [newline]Tram lines 1 and 15 provide direct access to the beach, city center, Delft, and Den Haag Central Station. You'll find shops and general stores conveniently located in the vicinity, open every day, with multiple supermarkets within walking distance for your daily needs. [newline] [newline]Distance to [newline] [newline]THUAS - 5 min biking distance [newline]Leiden University - 15 minute biking distance [newline]InHolland University - 20 min biking distance [newline]Albert Heijn Supermarket - 4 min biking distance [newline]Megastores shopping mall - 7 min biking distance [newline]Tram stop Broeksloot - 3 min walking distance [newline] [newline]", 'Samen op zoek naar een fijne woonruimte? In Almkerk hebben wij een woonruimte beschikbaar op basis van antikraak. Deze 2-kamerwoning van ca. 60 m² beschikt over een ruime woonkamer, aparte slaapkamer, functionele keuken, badkamer én een fijne tuin en is geschikt voor twee personen. [newline] [newline]Almkerk is een gezellig dorp met een rustige sfeer en een dorps karakter. De woning ligt op loopafstand van diverse winkels, supermarkten en andere voorzieningen. De snelweg is makkelijk te bereiken, wat zorgt voor een goede verbinding met de regio. In de buurt is er voldoende parkeergelegenheid.', 'Beste aankomend student, sinds 1988 is Huize Cook onderdeel van studerend Enschede. Een huis waar 6 mannen het waar maken om van hun studententijd de mooiste tijd te maken. [newline] [newline]Los het halen van onze studie, ondernemen we van alles met elkaar. Zoals huisavonden waar we beginnen met een door iemand georganiseerde activiteit en eindigen samen in de stad. Naast alles wat we doen als huisgenoten hebben we ook een unieke band met de oud bewoners. Zo organiseren we jaarlijks een kerstdiner waar alle oud bewoners uitgenodigd zijn. Hier komen de gekste verhalen boven tafel. Ook reizen we elk jaar met een grote groep een weekend naar een stad in Europa. Dit jaar zijn we bijvoorbeeld naar Gent gegaan! [newline] [newline]Buiten alles wat we met het huis doen maken we ook allemaal deel uit van een studentenvereniging waar we elke week te vinden zijn. [newline] [newline]Ons huis heeft naast de hechte band ook faciliteiten die het leven als student net iets aangenamer maken. Een vaatwasser en een wasmachine zijn onder studenten een aardige luxe, maar het toppunt is toch wel onze eigen tap naast het huis. Hier kunnen we altijd terecht zodat de woonkamer de volgende ochtend nog schoon genoeg is om rustig te ontbijten. [newline] [newline]Spreekt het jou aan om met 5 andere gasten de tijd van je leven te hebben? Ben jij iemand die houdt van een feestje en dingen ondernemen? Stuur dan een berichtje! [newline] [newline]Hulde!', "Sinds 1968 is Huize 't Pott al een studentenhuis. Met haar rijke geschiedenis die dateert van het jaar 1880, is Huize 't Pott niet alleen een studentenhuis, maar ook een iconisch huis voor de stad Enschede. [newline] [newline]Deze oude textielvilla is tegenwoordig het luxe onderkomen van 8 actieve en ondernemende studenten. Deze luxe is terug te vinden in onder andere de ruime kamers, die stuk voor stuk minstens 20 vierkante meter bedragen. Daarnaast beschikt ons huis over een royale woonkamer met eigen bar en open haard. Naast dat er een grote zithoek is, blijft er ook ruimte over voor onze pooltafel en bioscoop. Ook beschikken wij over twee keukens én twee badkamers en niet tenminste een jacuzzi. [newline] [newline]Naast al deze tastbare voordelen, is juist het ontastbare datgene wat dit huis zo mooi maakt. De sterke band die elke bewoner heeft met zowel elkaar als oud bewoners, maakt dat wij hier als een hechte vriendengroep leven. Zo ondernemen wij wekelijks leuke activiteiten met elkaar. Ook hebben wij verbanden, binnen en buiten Enschede, met vele huizen en disputen. Hierdoor zijn er vaak gezellige borrels. Naast dat er veel gezelligheid is, letten wij ook zeker op elkaars studie en stimuleren wij dit. Omdat wij elk onze eigen huistaak hebben, is verantwoordelijkheid iets wat zeker terug te vinden is op Huize 't Pott. [newline] [newline]Deze verantwoordelijkheid is ieder jaar ook terug te vinden in het grootste studenten huisgala van Nederland, 't Pott Gala. Dit traditionele gala organiseren wij volledig zelf, samen met de hulp van oud bewoners, in en rondom ons huis. Dit resulteert elk jaar weer in een prachtige avond, voor onszelf, maar ook voor de honderden bezoekers. [newline] [newline]Wil jij, net als wij, het meeste uit jouw studententijd halen en een band creëren voor het leven? Stuur ons dan vooral een bericht en dan zien wij jou misschien terug onder het genot van een koffietje of koude Heineken. [newline] [newline]HopHop!", 'This is ideal for a student or working student and very close to Metro Station. This is very close to all kinds of shops and very good connectivity.', 'Are you looking for a cozy living space in an authentic, peaceful location? This room is perfect for you! [newline] [newline]Located in a beautiful monumental building, this room offers a unique and inviting atmosphere. With a bus stop right outside and the train station just a 10-minute walk away, you can reach Maastricht in only 20 minutes. [newline] [newline]You will share the common areas with friendly housemates, creating a warm and relaxed environment. Plus, you’ll have access to a sunny terrace—ideal for unwinding after a busy day. [newline] [newline]Contact us soon to schedule a viewing and make this lovely room your new home!', 'Furnished room available in centrum from Eindhoven. There is a busstop next to the room wich goes directly to ASML and central station. The supermarket is within 300meters (Lidl, Albert Heijn and Albert Heijn XL). [newline] [newline]Its an international student house. There is a 1 month of deposit and contract goes trough mediator.']
image_cnt (int64, 22 distinct): ['10', '4', '6', '5', '7', '8', '9', '3', '2', '11']
img_0 (object, 1654 distinct, 0.3% missing): ['1845791/5.png', '1889831/5.png', '1918797/3.png', '1939742/3.png', '2045983/5.png', '2051047/3.png', '2062500/5.png', '2100669/5.png', '2102093/0.png', '2107186/3.png']
img_1 (object, 1635 distinct, 1.4% missing): ['1845791/3.png', '1889831/3.png', '1918797/1.png', '1939742/1.png', '2045983/3.png', '2051047/1.png', '2062500/3.png', '2100669/3.png', '2107186/1.png', '2110232/3.png']
img_2 (object, 1594 distinct, 3.9% missing): ['1845791/8.png', '1889831/1.png', '1918797/2.png', '1939742/2.png', '2045983/8.png', '2051047/2.png', '2062500/8.png', '2100669/1.png', '2107186/2.png', '2110232/8.png']
'''

IMG_RAW = "link"
def load_df(dir_path: str) -> DataFrame:
    main_dir = join(dir_path, MAIN_DIR)
    df = load_csv(main_dir, "listings_df.csv")
    handle_images(df, dir_path=dir_path)
    return df

def handle_images(df: DataFrame, dir_path: str):
    df[IMG_RAW] = df[IMG_RAW].apply(lambda i: _find_images_in_dir(dir_path=join(dir_path, IMAGE_FOLDER), prefix=str(i)))
    df['image_cnt'] = df[IMG_RAW].apply(lambda i: len(i))
    df['img_0'] = df[IMG_RAW].apply(lambda i: i[0] if len(i) > 0 else None)
    df['img_1'] = df[IMG_RAW].apply(lambda i: i[1] if len(i) > 1 else None)
    df['img_2'] = df[IMG_RAW].apply(lambda i: i[2] if len(i) > 2 else None)
    df.drop(columns=[IMG_RAW], inplace=True)
    return df


def _find_images_in_dir(dir_path: str, prefix: str) -> List[str]:
    full_path = join(dir_path, prefix)
    if not exists(full_path):
        return []
    return [join(prefix, img) for img in os.listdir(full_path)]

LABEL_NAME = ""
IMAGE_FEATURE_NAME = ""


CONTEXT = ""
TARGET = CuratedTarget(raw_name='furnish_type', task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = [
    # Constant
    "viewing_possible", "landlord_rented_before_n_times", "landlord_account_age"]
TEXT_FEATURES = [CuratedFeature(raw_name=f, feat_type=FeatureType.TEXT) for f in [
    'additionals', 'age_range', 'description', 'from_date', 'location', 'time_of_checking', 'to_date']]
FEATURES = [CuratedFeature(raw_name=f"img_{i}", feat_type=FeatureType.IMAGE) for i in range(3)] + TEXT_FEATURES
MAIN_DIR = "listings_data_11-07-2025"
IMAGE_FOLDER = join(MAIN_DIR, "listings_images")
LOADING_FUNC = load_df
