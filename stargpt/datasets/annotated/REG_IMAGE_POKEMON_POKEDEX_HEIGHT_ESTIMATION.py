from os.path import join, exists
from typing import Optional

from pandas import DataFrame

from tabstar2.utils.datasets import load_csv
from tabstar_paper.datasets.curation_objects import CuratedTarget, CuratedFeature
from tabstar_paper.datasets.objects import SupervisedTask, FeatureType


'''
Dataset Name: divyanshusingh369/complete-pokemon-library-32k-images-and-csv/
====
Examples: 1214
====
URL: https://www.kaggle.com/divyanshusingh369/complete-pokemon-library-32k-images-and-csv
====
Description:
title: Dataset of 32000 Pokemon Images & CSV, JSON
subtitle: Discover the Ultimate Pokémon Collection: 32K Images Await Your Exploration!!!
keywords: ['video games', 'computer science', 'computer vision', 'deep learning', 'recommender systems']
licenses: [{'name': 'apache-2.0'}]
description: If you like this dataset then please share your thoughts...

😄**[Support this dataset by clicking me 😺](https://www.paypal.com/paypalme/divyanshu3690)**😄

This dataset includes two directories with a collection of approximately 32,000 Pokémon images, along with two important files: a CSV file and a JSON file. The CSV file contains various details about each Pokémon, such as its name, type, species, height, weight, abilities, EV yield, catch rate, base friendship, base exp, growth rate, egg groups, gender, egg cycles, and base stats.

**CSV and JSON File Descriptors**

1. **Pokemon**: Name of the Pokémon.
2. **Type**: One or dual type determining weaknesses or resistances to attacks.
3. **Species**: Identifies the Pokémon based on defining biological characteristics.
4. **Height**: Height of each Pokémon.
5. **Weight**: Weight of each Pokémon.
6. **Abilities**: Special attributes aiding Pokémon in battle, introduced in Generation 3.
7. **EV Yield**: Stats gained by defeating specific Pokémon.
8. **Catch Rate**: Chances of catching a Pokémon with a Poké Ball.
9. **Base Friendship**: Default friendship value when encountering a Pokémon.
10. **Base Exp**: EXP yield when defeating a Pokémon at level 1.
11. **Growth Rate**: Amount of EXP needed for leveling up.
12. **Egg Groups**: Classification used in Pokémon breeding.
13. **Gender**: Chance of Pokémon being male or female.
14. **Egg Cycles**: Time unit for hatching Pokémon eggs.
15. **Base Stats (HP, Attack, Defense, Special Attack, Special Defense, Speed)**: Determine Pokémon strengths and weaknesses.

**Pokemon Dataset Directory**

This directory contains 973 Pokémon across all generations from 1 to 9, totaling 29,534 sprite images. Images may vary in dimensions, and animated GIFs are included for each Pokémon.

**Pokemon Images DB Directory**

In the Project Images DB directory, you'll find images of 1187 Pokémon, each accompanied by an original image and an image with the background removed. These images are ideal for projects like Pokémon Library, where you can display the images in your project just like the examples provided below.

![Pokemon Library Project Display](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F9519298%2Fe4750c0e9b93719e083aad63e9fabb70%2Fnationaldex.jpg?generation=1711662453395190&alt=media)

![Pokemon Library Project Display 2](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F9519298%2Faf62ce7c03756cdc44221b7dbbaba31f%2FFMo-WBjaMAUuPHE2.jpeg?generation=1711662948388237&alt=media)

The dataset is sourced from the [PokémonDB](https://pokemondb.net/pokedex/all) database, ensuring reliability and accuracy.

The images are sourced from Google.

**Sources**

All data and images are collected from the [PokémonDB](https://pokemondb.net/pokedex/all) database, maintaining authenticity and credibility. Additionally, a JSON file is provided for simplicity.

**If you find this dataset valuable for your projects, consider making a contribution of any amount to my PayPal Account. Even a small donation of $10 keeps me motivated to create more amazing datasets like this. Your support helps fuel further innovation and development. Additionally, feel free to leave comments about your experience with this dataset and contribute by applying it to machine learning algorithms in your Kaggle Notebook. Together, we can continue to drive progress and make meaningful advancements in the field.**

Paypal Account - **https://www.paypal.com/paypalme/divyanshu3690**

I hope you will learn something great from this dataset and will make amazing **Projects** by using this dataset.
====
Target Variable: Weight (float64, 526 distinct): ['0.3', '1.0', '120.0', '6.5', '8.0', '30.0', '4.0', '15.0', '5.0', '40.0']
====
Features:

Pokemon (object, 1214 distinct): ['Abomasnow', 'Mega Abomasnow', 'Abra', 'Absol', 'Mega Absol', 'Accelgor', 'Aegislash Shield Forme', 'Aegislash Blade Forme', 'Aerodactyl', 'Mega Aerodactyl']
Type (object, 221 distinct): ['Normal', 'Water', 'Psychic', 'Grass', 'Fire', 'Electric', 'Normal, Flying', 'Fighting', 'Bug', 'Ice']
Species (object, 733 distinct): ['Paradox Pokémon', 'Mouse Pokémon', 'Dragon Pokémon', 'Fox Pokémon', 'Bagworm Pokémon', 'Pumpkin Pokémon', 'Flame Pokémon', 'Fossil Pokémon', 'Puppy Pokémon', 'Mushroom Pokémon']
Height (object, 62 distinct): ['0.6 m (2′00″)', '0.3 m (1′00″)', '0.4 m (1′04″)', '0.5 m (1′08″)', '1.0 m (3′03″)', '1.2 m (3′11″)', '1.5 m (4′11″)', '0.7 m (2′04″)', '0.8 m (2′07″)', '1.4 m (4′07″)']
Abilities (object, 704 distinct): ['1. Levitate', '1. Beast Boost', '1. Protosynthesis', '1. Quark Drive', '1. Pickup, 2. Frisk, Insomnia (hidden ability)', '1. Pressure', '1. Pressure, Telepathy (hidden ability)', '1. Shed Skin', '1. Justified', '1. Poison Point, 2. Rivalry, Hustle (hidden ability)']
EV Yield (object, 50 distinct): ['2 Attack', '1 Attack', '1 Speed', '2 Speed', '3 Attack', '3 Sp. Atk', '2 Defense', '2 Sp. Atk', '1 Defense', '1 HP']
Catch Rate (object, 37 distinct): ['45 (5.9% with PokéBall, full HP)', '3 (0.4% with PokéBall, full HP)', '190 (24.8% with PokéBall, full HP)', '255 (33.3% with PokéBall, full HP)', '75 (9.8% with PokéBall, full HP)', '120 (15.7% with PokéBall, full HP)', '60 (7.8% with PokéBall, full HP)', '90 (11.8% with PokéBall, full HP)', '30 (3.9% with PokéBall, full HP)', '25 (3.3% with PokéBall, full HP)']
Base Friendship (object, 9 distinct): ['50 (normal)', '35 (lower than normal)', '0 (lower than normal)', '—', '100 (higher than normal)', '140 (higher than normal)', '90 (higher than normal)', '20 (lower than normal)', '70 (higher than normal)']
Base Exp (float64, 202 distinct, 1.9% missing): ['60.0', '62.0', '175.0', '61.0', '172.0', '142.0', '168.0', '270.0', '173.0', '170.0']
Growth Rate (object, 6 distinct): ['Medium Fast', 'Slow', 'Medium Slow', 'Fast', 'Erratic', 'Fluctuating']
Egg Groups (object, 61 distinct): ['Field', 'Undiscovered', 'Bug', 'Mineral', 'Flying', 'Amorphous', 'Human-Like', 'Grass', 'Water 2', 'Field, Water 1']
Gender (object, 8 distinct): ['50% male, 50% female', 'Genderless', '87.5% male, 12.5% female', '0% male, 100% female', '100% male, 0% female', '25% male, 75% female', '75% male, 25% female', '12.5% male, 87.5% female']
Egg Cycles (object, 12 distinct): ['20 (4,884–5,140 steps)', '15 (3,599–3,855 steps)', '120 (30,584–30,840 steps)', '25 (6,169–6,425 steps)', '40 (10,024–10,280 steps)', '30 (7,454–7,710 steps)', '35 (8,739–8,995 steps)', '10 (2,314–2,570 steps)', '—', '50 (12,594–12,850 steps)']
HP Base (int64, 109 distinct): ['60', '70', '50', '80', '75', '65', '40', '100', '45', '90']
HP Min (int64, 109 distinct): ['230', '250', '210', '270', '260', '240', '190', '310', '200', '290']
HP Max (int64, 109 distinct): ['324', '344', '304', '364', '354', '334', '284', '404', '294', '384']
Attack Base (int64, 126 distinct): ['100', '65', '80', '75', '85', '60', '70', '55', '50', '95']
Attack Min (int64, 126 distinct): ['184', '121', '148', '139', '157', '112', '130', '103', '94', '175']
Attack Max (int64, 126 distinct): ['328', '251', '284', '273', '295', '240', '262', '229', '218', '317']
Defense Base (int64, 114 distinct): ['70', '60', '50', '80', '65', '90', '40', '100', '45', '55']
Defense Min (int64, 114 distinct): ['130', '112', '94', '148', '121', '166', '76', '184', '85', '103']
Defense Max (int64, 114 distinct): ['262', '240', '218', '284', '251', '306', '196', '328', '207', '229']
Special Attack Base (int64, 127 distinct): ['40', '60', '50', '65', '55', '45', '70', '80', '95', '30']
Special Attack Min (int64, 127 distinct): ['76', '112', '94', '121', '103', '85', '130', '148', '175', '58']
Special Attack Max (int64, 127 distinct): ['196', '240', '218', '251', '229', '207', '262', '284', '317', '174']
Special Defense Base (int64, 106 distinct): ['80', '70', '50', '60', '65', '55', '75', '90', '45', '40']
Special Defense Min (int64, 106 distinct): ['148', '130', '94', '112', '121', '103', '139', '166', '85', '76']
Special Defense Max (int64, 106 distinct): ['284', '262', '218', '240', '251', '229', '273', '306', '207', '196']
Speed Base (int64, 127 distinct): ['50', '60', '65', '70', '30', '90', '85', '45', '40', '100']
Speed Min (int64, 127 distinct): ['94', '112', '121', '130', '58', '166', '157', '85', '76', '184']
Speed Max (int64, 127 distinct): ['218', '240', '251', '262', '174', '306', '295', '207', '196', '328']
pokemon_image (object, 1077 distinct, 11.3% missing): ['Abomasnow/Abomasnow.png', 'Mega Abomasnow/Mega Abomasnow.png', 'Abra/Abra.png', 'Absol/Absol.png', 'Mega Absol/Mega Absol.png', 'Accelgor/Accelgor.png', 'Aerodactyl/Aerodactyl.png', 'Mega Aerodactyl/Mega Aerodactyl.png', 'Aggron/Aggron.png', 'Mega Aggron/Mega Aggron.png']
pokemon_new_image (object, 1076 distinct, 11.4% missing): ['Abomasnow/Abomasnow_new.png', 'Mega Abomasnow/Mega Abomasnow_new.png', 'Abra/Abra_new.png', 'Absol/Absol_new.png', 'Mega Absol/Mega Absol_new.png', 'Accelgor/Accelgor_new.png', 'Aerodactyl/Aerodactyl_new.png', 'Mega Aerodactyl/Mega Aerodactyl_new.png', 'Aggron/Aggron_new.png', 'Mega Aggron/Mega Aggron_new.png']
'''


OLD_POKEMON_IMG = "pokemon_image"
NEW_POKEMON_IMG = "pokemon_new_image"



def load_df(dir_path: str) -> DataFrame:
    df = load_csv(dir_path, "pokemonDB_dataset.csv")
    pokemon_names = sorted(set(df['Pokemon']))
    assert len(df) == len(pokemon_names), "Pokemon names not unique"
    df[OLD_POKEMON_IMG] = df['Pokemon'].apply(lambda p: pokemon_path_if_exists(dir_path=dir_path, pokemon=p, is_new=False))
    df[NEW_POKEMON_IMG] = df['Pokemon'].apply(lambda p: pokemon_path_if_exists(dir_path=dir_path, pokemon=p, is_new=True))
    df['Base Exp'] = df['Base Exp'].apply(parse_base_exp)
    df['Height'] = df['Height'].apply(parse_height)
    df['Weight'] = df['Weight'].apply(parse_weight)
    return df


def pokemon_path_if_exists(dir_path: str, pokemon: str, is_new: bool) -> str | None:
    image_path = join(dir_path, IMAGE_FOLDER)
    filename = f"{pokemon}_new.png" if is_new else f"{pokemon}.png"
    pokemon_filepath = join(pokemon, filename)
    full_path = join(image_path, pokemon_filepath)
    if exists(full_path):
        return pokemon_filepath
    else:
        return None


def parse_height(height: str) -> float:
    # '0.6 m (2′00″)', '0.3 m (1′00″)', '0.4 m (1′04″)', '0.5 m (1′08″)'
    if not height.count('m') == 1:
        raise ValueError(f"Height is not in the correct format: {height}")
    h, _ = height.split('m')
    return float(h.strip())

def parse_weight(weight: str) -> float | None:
    # Weight looks like: '135.5 kg (298.7 lbs)'
    if weight == "—":
        return None
    w, _ = weight.split('kg')
    return float(w.strip())


def parse_base_exp(base_exp: str) -> int | None:
    if base_exp == "—":
        return None
    return int(base_exp)


CONTEXT = "Pokemon Pokedex with images"
TARGET = CuratedTarget(raw_name="Height", task_type=SupervisedTask.REGRESSION)
FEATURES = [CuratedFeature(raw_name=OLD_POKEMON_IMG, feat_type=FeatureType.IMAGE),
            CuratedFeature(raw_name=NEW_POKEMON_IMG, feat_type=FeatureType.IMAGE),
            CuratedFeature(raw_name="Pokemon", feat_type=FeatureType.TEXT),
            CuratedFeature(raw_name="Type", feat_type=FeatureType.TEXT),
            CuratedFeature(raw_name="Species", feat_type=FeatureType.TEXT),
            CuratedFeature(raw_name="Abilities", feat_type=FeatureType.TEXT),]
IMAGE_FOLDER = "Pokemon Images DB/Pokemon Images DB"
LOADING_FUNC = load_df
