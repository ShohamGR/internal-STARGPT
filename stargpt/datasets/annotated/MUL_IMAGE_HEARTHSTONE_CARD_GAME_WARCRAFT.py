import os
from os.path import join, exists

import pandas as pd
from pandas import DataFrame

from tabstar2.utils.datasets import load_csv
from tabstar2.utils.images import download_url_dataset, unzip_url_dataset
from tabstar_paper.datasets.curation_objects import CuratedTarget, CuratedFeature
from tabstar_paper.datasets.objects import SupervisedTask, FeatureType

'''
Dataset Name: MUL_IMAGE_HEARTHSTONE_CARD_GAME_WARCRAFT
====
Examples: 10700
====
URL: https://figshare.com/ndownloader/files/38561075
====
Description: 

# Similar to: https://www.kaggle.com/jeradrose/hearthstone-cards

Hearthstone Cards Dataset
Describes collectable cards from the game Hearthstone, with features covering gameplay stats, categorical traits, and
rich textual descriptions. The prediction target is the player class to which a card belongs (10 categories), with a
notable class imbalance. Neutral cards dominate, while other classes have significantly fewer samples. This natural
imbalance reflects the game design and poses a realistic challenge for classification tasks. Text fields like text (card
effects) and flavor (lore snippets) provide valuable semantic signals, making this dataset well-suited to benchmark how
textual features can improve tabular model performance, especially for minority classes.


Description: 
This dataset contains data for the entire collection of cards for Hearthstone, the popular online card game by Blizzard. Launching to the public on March 11, 2011 after being under development for almost 5 years, Hearthstone has gained popularity as a freemium game, launching into eSports across the globe, and the source of many Twitch channels.

The data in this dataset was extracted from hearthstonejson.com, and the documentation for all the data can be found on the cards.json documentation page.

The original data was extracted from the actual card data files used in the game, so all of the data should be here, enabling explorations like:

Card strengths and weaknesses
Card strengths relative to cost and rarity
Comparisons across player classes, bosses, and sets
Whether a set of optimal cards can be determined per class
The cards can be explored in one of four ways:

cards.json: The raw JSON pulled from hearthstonejson.com
cards_flat.csv: A flat CSV containing a row for each card, and any n:m data stored as arrays in single fields
database.sqlite: A SQLite database containing relational data of the cards
cards.csv, mechanics.csv, dust_costs.csv, play_requirements.csv, and entourages.csv: the normalized data in CSV format.
This dataset will be updated as new releases and expansions are made to Hearthstone.

Currently, any localized string values are in en-us, but I may look into adding other languages if the demand seems to be there.

====
Target Variable: cardClass (object, 13 distinct): ['NEUTRAL', 'NONE_cardClass', 'DRUID', 'WARLOCK', 'HUNTER', 'MAGE', 'PALADIN', 'WARRIOR', 'ROGUE', 'SHAMAN']
====
Features:

health (float64, 97 distinct, 34.8% missing): ['4.0', '3.0', '2.0', '1.0', '5.0', '6.0', '8.0', '7.0', '10.0', '15.0']
name (object, 7474 distinct): ['???', 'Transfer Student', 'Jade Golem', 'The Coin', 'Treant', "Bru'kan", 'Rokara', "Ozumat's Tentacle", 'Druid of the Claw', 'Xyrella']
set (object, 40 distinct): ['LETTUCE', 'TB', 'BATTLEGROUNDS', 'THE_SUNKEN_CITY', 'THE_BARRENS', 'VANILLA', 'DARKMOON_FAIRE', 'ALTERAC_VALLEY', 'STORMWIND', 'EXPERT1']
type (object, 4 distinct): ['MINION', 'SPELL', 'WEAPON', 'LOCATION']
attack (float64, 36 distinct, 31.8% missing): ['2.0', '3.0', '4.0', '1.0', '5.0', '0.0', '6.0', '8.0', '10.0', '7.0']
cost (int64, 19 distinct): ['3', '4', '1', '2', '0', '5', '6', '7', '8', '10']
rarity (object, 5 distinct): ['FREE', 'COMMON', 'RARE', 'LEGENDARY', 'EPIC']
artist (object, 407 distinct, 45.6% missing): ['Zoltan Boros', 'Konstantin Turovec', 'Matt Dixon', 'James Ryman', 'Alex Horley Orlandelli', 'Anton Zemskov', 'Jim Nelson', 'Arthur Bozonnet', 'Ivan Fomin', 'Dave Allsop']
spellSchool (object, 7 distinct, 89.8% missing): ['NATURE', 'SHADOW', 'HOLY', 'FIRE', 'ARCANE', 'FROST', 'FEL']
text (object, 7150 distinct, 10.6% missing): ['<b>Taunt</b>', '<b>Rush</b>', '<b>???</b>', '<b>Dormant</b>', '[x]<b>Lorebook</b>\n<i>Click on this to read it.</i>\nLasts one turn.', '<b>Charge</b>', '<b>Stealth</b>', '<b>Windfury</b>', '<b>Spell Damage +1</b>', 'Gain 1 Mana Crystal this turn only.']
mechanics (object, 359 distinct, 41.7% missing): ["['BATTLECRY']", "['TRIGGER_VISUAL']", "['DEATHRATTLE']", "['TAUNT']", "['DUNGEON_PASSIVE_BUFF']", "['AURA']", "['RUSH']", "['DISCOVER']", "['SECRET']", "['BATTLECRY', 'TAUNT']"]
race (object, 34 distinct, 68.7% missing): ['BEAST', 'ELEMENTAL', 'DEMON', 'MECHANICAL', 'DRAGON', 'MURLOC', 'PIRATE', 'NAGA', 'ORC', 'QUILBOAR']
durability (float64, 8 distinct, 96.8% missing): ['2.0', '3.0', '4.0', '1.0', '5.0', '8.0', '6.0', '10.0']
overload (float64, 6 distinct, 99.3% missing): ['2.0', '1.0', '3.0', '4.0', '10.0', '5.0']
spellDamage (float64, 5 distinct, 99.3% missing): ['1.0', '2.0', '3.0', '5.0', '4.0']
Image Path (object, 10700 distinct): ['train_images/AT_004.jpg', 'train_images/Story_06_Meryl.jpg', 'train_images/EX1_275.jpg', 'train_images/CS2_029.jpg', 'train_images/TSC_029t2.jpg', 'train_images/AV_284.jpg', 'train_images/DMF_104.jpg', 'train_images/FB_LK002.jpg', 'train_images/CORE_CS2_033.jpg', 'train_images/REV_602.jpg']
'''

CARD_CLASS = "cardClass"

def load_df(dir_path: str) -> DataFrame:
    if not exists(dir_path):
        _download_and_unzip(dir_path)
    df = _collect_split_datasets(dir_path=dir_path)
    df = take_top_10_most_common_card_classes(df)
    return df

def take_top_10_most_common_card_classes(df: DataFrame) -> DataFrame:
    card_class_counts = df[CARD_CLASS].value_counts()
    top_10_card_classes = card_class_counts.head(10).index
    df = df[df[CARD_CLASS].isin(top_10_card_classes)]
    return df

def _download_and_unzip(dir_path: str):
    zip_path = download_url_dataset(url="https://figshare.com/ndownloader/files/38561075", path="HearthStoneCards")
    unzip_url_dataset(src_path=zip_path, dst_path=dir_path)
    for split in ['train', 'dev', 'test']:
        split_zip_name = join(dir_path, IMAGE_FOLDER, f"{split}_images.zip")
        unzip_url_dataset(src_path=split_zip_name)
        os.remove(split_zip_name)

def _collect_split_datasets(dir_path: str) -> DataFrame:
    main_dir = join(dir_path, IMAGE_FOLDER)
    dfs = []
    for split in ['train', 'dev', 'test']:
        split_df = load_csv(dir_path=main_dir, filename=f"{split}.csv")
        dfs.append(split_df)
    df = pd.concat(dfs)
    return df

TARGET = CuratedTarget(raw_name=CARD_CLASS, task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = ['id', 'collectible']
TEXT_FEATURES = [CuratedFeature(raw_name=name, feat_type=FeatureType.TEXT) for name in ['artist', 'mechanics', 'name', 'text']]
FEATURES = [CuratedFeature(raw_name='Image Path', feat_type=FeatureType.IMAGE),] + TEXT_FEATURES
IMAGE_FOLDER = "Hearthstone-All-cardClass"
LOADING_FUNC = load_df