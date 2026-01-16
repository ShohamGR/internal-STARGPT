from doctest import DocFileSuite
from inspect import isasyncgen
from ntpath import exists
import os
from os.path import join
from typing import Optional

from pandas import DataFrame, read_csv

from tabstar2.utils.datasets import load_csv
from tabstar_paper.datasets.curation_objects import CuratedTarget, CuratedFeature
from tabstar_paper.datasets.objects import SupervisedTask, FeatureType


IMAGE_FEATURE_NAME = "token_id"

'''
Dataset Name: harrywang/crypto-coven/
====
Examples: 9764
====
URL: https://www.kaggle.com/harrywang/crypto-coven
====
Description: 
Crypto Coven
When data science meets NFT

About Dataset
This dataset contains information about the 9761 witches from the Crypto Coven NFT project (https://www.cryptocoven.xyz/) collected using OpenSea API. The full data returned by the API is provided in witches_full.csv and a subset of the data is chosen by me and shared in witches.csv. The folder 'witch_images' includes the images of each witch in three different sizes.

I briefly describe the data in the witches.csv below:

id: the id of the witch
num_sales: number of sales in the past (till 4/21/2022 the day I collected the data)
name: the name of the witch
description: the description of the witch
external_link: the link to the official page for the witch
permalink: the OpenSea link for the witch
token_metadata: the metadata JSON file about the witch
token_id: the token_id of the NFT
owner.user.username: the user name of the current owner
owner.address: the wallet address of the current owner
last_sale.total_price: the price of the last sale in gwei. Note that the unit here is gwei (giga and wei) and 1 ether = 1 billion gwei (18 zeros)
last_sale.payment_token.usd_price: the USD price of 1 ether (ETH) for the last sale
last_sale.transaction.timestamp: the timestamp of the last sale
properties: there are 32 properties of each witch covering the different design elements of each witch, such as Skin Tone, Eyebrows, Body Shape, etc.
witches_full.csv is the full data provided by the OpenSea API, such as https://api.opensea.io/api/v1/asset/0x5180db8f5c931aae63c74266b211f580155ecac8/50. I just simply flattened the JSON returned by the API.
====
Target Variable: num_sales (object, 5 distinct): ['0', '1', '2', '3', '4+']
====
Features:

name (object, 9761 distinct): ['honeydew, the delicate beach', 'alizarin, the covert crucible', 'ruby, the mauve vase', 'kiwi, the mercurial dream', 'hyacinth conch', 'imperatrix francium', 'balsa, the wicked vault', 'alexandrite, the luscious circlet', 'antlia abacus', 'violet the dreamy']
description (object, 9761 distinct): ['You are a WITCH made of sunshine. You fence your garden with the ribs of a sun-bleached whale. Your magic spawns from... your hat? The cards have spoken to you since you were small. FILL YOUR MIND WITH INCANTATIONS!', 'You are a WITCH made of daydreams. Your baked goods are conniving concoctions known the land over. Your magic spawns from yew bark and yodeling. Ghosts clutter your castle in heaping mounds of whispers.  MAKE WAY!', 'You are a WITCH made of predator teeth. Your consciousness is heavy and vast. Your magic spawns from a kitten’s first steps. Nations rise and fall at your whim. GET VOLCANIC ASH IN YOUR HAIR!', 'You are a WITCH with eyes that hold oceans. You are an explorer of the far-flung unknown, a messenger from the edge. Your magic spawns from vitamin D. Your untangle your fate from the follies of the world. IT’S TIME TO SCREAM!', 'You are a WITCH bold beyond reckoning. You whisper your secrets to mummified heads. Your magic spawns from the sweat on your back. You see wonders and wastelands in the leaves of your tea. RUN INTO THE WATER!', 'You are a WITCH whose existence is secret. You carve gemstones with sigils and figures. Your magic spawns from a cloud of pepper spray. You know all the most scandalous secrets of the dead. MAKE WAY!', 'You are a WITCH made of greening copper. You tattoo your own skin with wasps dipped in squid ink. Your magic spawns from the depths of the earth. You carry the soul of your sister in a ring upon your finger. IT’S TIME TO DANCE!', 'You are a WITCH woven from the strings of a windchime. You make your own jewelry from trinkets and dried hearts. Your magic spawns from the warmth of a good book with ten very attractive people on the cover. Illusions drip from your gold-dipped combs. RUTHLESS KINDNESS!', 'You are a WITCH made of fresh-cut grass. You collect whispers embedded deep within the forest. Your magic spawns from leaves curling inwards for winter. You sleep upon stacks of books, dream of the intricacies of the universe. RUN BACK TO THE BEGINNING!', 'You are a WITCH with unspeakable talent. You dance among flames, do-si-do and jitterbug in fire. Your magic spawns from a fawn’s first steps. You see visions of the future in the eyes of your loved ones. FIAT LUX!']
token_id (object, 9761 distinct): ['3507/3507_os.png', '7811/7811_os.png', '6148/6148_os.png', '6503/6503_os.png', '6505/6505_os.png', '6506/6506_os.png', '6507/6507_os.png', '6508/6508_os.png', '6509/6509_os.png', '6510/6510_os.png']
Wonder (int64, 11 distinct): ['9', '3', '5', '6', '4', '7', '8', '2', '1', '0']
Skin Tone (object, 7 distinct): ['Twilight', 'Dawn', 'Day', 'Sunset', 'Night', 'Dusk', 'Midnight']
Rising Sign (object, 12 distinct): ['Libra', 'Leo', 'Sagittarius', 'Scorpio', 'Taurus', 'Virgo', 'Cancer', 'Pisces', 'Aries', 'Aquarius']
Eyebrows (object, 16 distinct, 1.3% missing): ['Strong (Black)', 'Bushy (Brown)', 'Thin Arched (Grey)', 'Medium Flat (Black)', 'Arched (Brown)', 'Bushy (White)', 'Round (Navy)', 'Pencil (White)', 'Pierced (White)', 'Pierced Pyramid Stud (Grey)']
Wisdom (int64, 11 distinct): ['9', '4', '3', '7', '8', '5', '6', '1', '2', '0']
Body Shape (object, 3 distinct): ['Lithe', 'Chiseled', 'Soft']
Moon Sign (object, 12 distinct): ['Cancer', 'Libra', 'Capricorn', 'Sagittarius', 'Taurus', 'Scorpio', 'Gemini', 'Aquarius', 'Leo', 'Aries']
Will (int64, 11 distinct): ['9', '7', '3', '4', '8', '6', '5', '2', '0', '1']
Hair Color (object, 32 distinct, 2.0% missing): ['Silver', 'Steel', 'Platinum', 'Ash', 'Mermaid', 'Auburn', 'Lilac', 'Lavender', 'Mauve', 'Navy']
Wit (int64, 11 distinct): ['7', '8', '6', '3', '4', '5', '1', '2', '9', '0']
Wiles (int64, 10 distinct): ['3', '4', '8', '5', '6', '7', '9', '2', '0', '1']
Necklace (object, 26 distinct, 76.3% missing): ['Moon Necklace (Silver)', 'Studded Collar (Silver)', 'Creepy Yeha Pearls (Pearl)', 'Thread Necklace (Natural)', 'Metal Collar (Silver)', 'Ethereum Thread (Natural)', 'Moon Choker (Silver)', 'Eclectic Choker Stack (Green)', 'Eclectic Choker Stack (Red)', 'Natural Dried Orange Stack (Natural)']
Sun Sign (object, 12 distinct): ['Virgo', 'Leo', 'Scorpio', 'Aquarius', 'Gemini', 'Taurus', 'Aries', 'Cancer', 'Sagittarius', 'Libra']
Eye Style (object, 29 distinct, 0.0% missing): ['Circle Lens', 'Hooded Side', 'Sunset Almond', 'Babylash', 'Smudged Snake', 'Alert', 'Smudged', 'Reverse Cateye', 'Hypnotic', 'Blank']
Eye Color (object, 36 distinct, 0.0% missing): ['Grey', 'Gold', 'Pink', 'Glo', 'Red', 'Green', 'Purple', 'Amber', 'Violet', 'Hazel']
Mouth (object, 30 distinct, 0.0% missing): ['Parted (Beige)', 'Parted (Black)', 'Plump (Natural)', 'Bow (Red)', 'Bow (Purple)', 'Round (Dark)', 'Bow (Coral)', 'Witchy Pout (Black)', 'Bow (Dusk)', 'Glossy Medusa (Natural)']
Hat (object, 26 distinct, 72.7% missing): ['Witch', 'Hoodlondon Hathor (Gold)', 'Lined (Black)', 'Mushroom', 'Arcane Stars (Black)', 'Chic Disk (Black)', 'Floral Lacewing', 'Young Floral Horns', 'Moon Moth (Black)', 'Dita Disk (Black)']
Archetype of Power (object, 11 distinct): ['Occultist', 'Hag', 'Mage', 'Necromancer', 'Seer', 'Enchantress', 'Witch of Woe', 'Witch of Will', 'Witch of Wisdom', 'Witch of Wonder']
Woe (int64, 11 distinct): ['9', '7', '4', '3', '6', '5', '8', '2', '0', '1']
Hair (Front) (object, 28 distinct): ['Xuannu', 'Aradia Short', 'Aletheia', 'Curtain', 'Curl Cloud', 'Aradia', 'Baby Drill Curls', 'Keridwen', 'Dramatic Swoop', 'Nyx']
Top (object, 48 distinct, 0.0% missing): ['Straps (Black)', 'Mugler Collarbone Cutout (Black)', 'Mock Neck (Black)', 'Striped Shroud (Umber)', 'Sheer Striped (Black)', 'Ragged Shawl (Black)', 'Sheer Sleeves (Black)', 'Shroud (Lilac)', 'Leather Neck Piece (Black)', 'Shroud (Dusk)']
Hair (Back) (object, 26 distinct, 26.0% missing): ['Soft Braid', 'Aradia', 'Flowy Pony', 'Keridwen', 'Dramatic Swoop', 'High Blowout', 'Aletheia', 'Curl Cloud', 'Cute Pigtails', 'Soft Waves']
Background (object, 18 distinct): ['Pink', 'Plum', 'Lavender', 'Taupe', 'Sea', 'Moss', 'Peach', 'Rust', 'Solid (Rust)', 'Solid (Lavender)']
Face Markings (object, 61 distinct, 68.9% missing): ['Cheek Mole', 'Freckles', 'Nose Bridge Scar', 'Cheek and Chin Mole', 'Undereye Gems (White)', 'Eye of Hinoto', 'Mark of Moon Tears (White)', 'Horned Moon Markings (Black)', 'Moon Mark (White)', 'Mark of the Chained Moon (White)']
Facewear (object, 28 distinct, 64.3% missing): ['Bridge Piercing', 'Septum Ring (Turquoise)', 'Labret (Silver)', 'Jeweled Septum Ring (Blue)', 'Simple Septum Ring (Silver)', 'Sad Girl Tears', 'Simple Octagon Glasses (Silver)', 'Givenchy Nosering (Silver)', 'Swiss Dot Veil (Black)', 'Eyepatch (Rose)']
Hair Topper (object, 12 distinct, 73.9% missing): ['Bat Wings', 'Crown of Pearls', 'Maleficent Horns', 'Horns of Taurus', 'Aries Horns', 'Gentle Horns', 'Jeweled Antlers', 'Leafy Crown', 'Anemone Horns', 'Feathered Hairpins']
Back Item (object, 10 distinct, 50.5% missing): ['Monstera', 'Lily', 'Chinese Lantern', 'Scripted Circle (Blue)', 'Scripted Mage Aura (Purple)', 'Scripted Mage Aura (Blue)', 'Scripted Mage Aura (Gold)', 'Scripted Mage Aura (Green)', 'Mage Aura (Orange)', 'Arcane Circuitry Halo']
Earrings (object, 21 distinct, 70.8% missing): ['Basic Piercings (Silver)', 'Dangly Bones (Bone)', 'Pearly Skulls (Silver)', 'Beaded Tassel (Ruby)', 'Beaded Tassel (Emerald)', 'Howl Gems (Orange)', 'Howl Gems (Purple)', 'Beaded Tassel (Sapphire)', 'Ethereum Hoops (Silver)', 'Bone Hoops (Silver)']
Forehead Jewelry (object, 24 distinct, 88.8% missing): ['Ethereum Diadem (Silver)', 'Demon Third Eye (Black)', 'Darkened Demon Third Eye (Opal)', 'Darkened Demon Third Eye (Sapphire)', 'Darkened Demon Third Eye (Ruby)', 'Raised Third Eye (Red)', 'Darkened Demon Third Eye (Glo)', 'Raised Third Eye (Pink)', 'Punks Tiara (Gold)', 'Simple Circlet (Silver)']
Hair (Middle) (object, 4 distinct, 91.6% missing): ['Dark Lady', 'Cute Buns', 'Double Locs', 'Flower Buns']
Mask (object, 15 distinct, 94.9% missing): ['Moth Gauze (White)', 'Graceful Bat', 'Butterfly (Gold)', 'Butterfly (Emerald)', 'Butterfly (Rose Quartz)', 'Glowing Phosphorous Mask (Purple)', 'Gold Veil (Gold)', 'Glowing Phosphorous Mask (Aqua)', 'Butterfly Glasses (Cyan)', 'Mage Glass Moon (Purple)']
Outerwear (object, 6 distinct, 94.3% missing): ['Feather Robe (White)', 'Bone Corset (White)', 'Bone Corset (Black)', 'Feather Robe (Black)', 'Crochet (White)', 'School Uniform (Black)']

'''

def load_df(dir_path: str) -> DataFrame:
    df = load_csv(dir_path, "witches.csv")
    # full_df = load_csv(dir_path, "witches_full.csv")
    df[IMAGE_FEATURE_NAME] = df[IMAGE_FEATURE_NAME].apply(lambda x: _get_image_path(x, dir_path=dir_path))
    return df

def _group_num_sales(num_sales: int) -> int:
    """
    (Pdb) df['num_sales'].value_counts()
    num_sales
    0    4866
    1    2560
    2    1425
    3     587
    4     222
    5      65
    6      27
    7       7
    8       4
    9       1
    """
    if num_sales in [0, 1, 2, 3]:
        return str(num_sales)
    assert num_sales >= 4
    return "4+"


def _get_image_path(token_num: int, dir_path: str) -> Optional[str]:
    img_dir = join(dir_path, IMAGE_FOLDER)
    token_img_path = f"{token_num}/{token_num}_os.png"
    if not exists(join(img_dir, token_img_path)):
        return None
    return token_img_path


CONTEXT = ""
TARGET = CuratedTarget(raw_name='num_sales', task_type=SupervisedTask.MULTICLASS, processing_func=_group_num_sales)
COLS_TO_DROP = ['id', 
    # Irrelevant
    'owner.user.username', 'owner.address', 
    # Links to websites
    'external_link', 'permalink', 'token_metadata',
    # Leakage
    'last_sale.transaction.timestamp', 'last_sale.total_price', 'last_sale.payment_token.usd_price']
FEATURES = [CuratedFeature(raw_name=IMAGE_FEATURE_NAME, feat_type=FeatureType.IMAGE)]
IMAGE_FOLDER = "witch_images/witch_images"
LOADING_FUNC = load_df
