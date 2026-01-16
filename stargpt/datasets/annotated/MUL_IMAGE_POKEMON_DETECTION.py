import os
from os.path import join

from pandas import DataFrame

from tabstar_paper.datasets.curation_objects import CuratedTarget, CuratedFeature
from tabstar_paper.datasets.objects import SupervisedTask, FeatureType


'''
Dataset Name: chatrlichap/151-first-pokemons-images/
====
Examples: 7550
====
URL: https://www.kaggle.com/chatrlichap/151-first-pokemons-images
====
Description: 
This contains 151 folder for each first 151 Pokemons.
Each folder is named after its Pokemon.
Each folder contains 50 images scrapped from the web.

This dataset as been made for the purpose of a vision model that would be able to recognize any of the first 151 Pokemons, a draw, a Pokemon card, an fan-art, even from the movies or animes.
====
Target Variable: Pokemon Name (object, 151 distinct): ['vaporeon', 'omastar', 'wartortle', 'magmar', 'jigglypuff', 'meowth', 'cubone', 'horsea', 'slowpoke', 'shellder']
====
Features:

Pokemon (object, 7550 distinct): ['vaporeon/vaporeon_18.png', 'vaporeon/vaporeon_23.png', 'vaporeon/vaporeon_45.png', 'vaporeon/vaporeon_30.png', 'vaporeon/vaporeon_9.png', 'vaporeon/vaporeon_11.png', 'vaporeon/vaporeon_38.png', 'vaporeon/vaporeon_20.png', 'vaporeon/vaporeon_19.png', 'vaporeon/vaporeon_8.png']
'''


LABEL_NAME = "Pokemon Name"
IMAGE_FEATURE_NAME = "Pokemon"



def load_df(dir_path: str) -> DataFrame:
    df = load_pokemon_df(dir_path)
    df = take_10_similar_pokemon_types(df)
    return df

def load_pokemon_df(dir_path: str) -> DataFrame:
    ret = []
    for pokemon_name in os.listdir(dir_path):
        for pokemon_pic in os.listdir(join(dir_path, pokemon_name)):
            final_path = join(pokemon_name, pokemon_pic)
            ret.append({IMAGE_FEATURE_NAME: final_path, LABEL_NAME: pokemon_name})
    ret = DataFrame(ret)
    return ret

def take_10_similar_pokemon_types(df: DataFrame) -> DataFrame:
    """
    ['abra', 'aerodactyl', 'alakazam', 'arbok', 'arcanine', 'articuno', 'beedrill', 'bellsprout', 'blastoise', 'bulbasaur', 'butterfree', 'caterpie', 
    'chansey', 'charizard', 'charmander', 'charmeleon', 'clefable', 'clefairy', 'cloyster', 'cubone', 'dewgong', 'diglett', 'ditto', 'dodrio', 'doduo',
     'dragonair', 'dragonite', 'dratini', 'drowzee', 'dugtrio', 'eevee', 'ekans', 'electabuzz', 'electrode', 'exeggcute', 'exeggutor', 'farfetchd', 
     'fearow', 'flareon', 'gastly', 'gengar', 'geodude', 'gloom', 'golbat', 'goldeen', 'golduck', 'golem', 'graveler', 'grimer', 'growlithe', 'gyarados', 
     'haunter', 'hitmonchan', 'hitmonlee', 'horsea', 'hypno', 'ivysaur', 'jigglypuff', 'jolteon', 'jynx', 'kabuto', 'kabutops', 'kadabra', 'kakuna', 
     'kangaskhan', 'kingler', 'koffing', 'krabby', 'lapras', 'lickitung', 'machamp', 'machoke', 'machop', 'magikarp', 'magmar', 'magnemite', 'magneton', 
     'mankey', 'marowak', 'meowth', 'metapod', 'mew', 'mewtwo', 'moltres', 'mr-mime', 'muk', 'nidoking', 'nidoqueen', 'nidoran-f', 'nidoran-m', 
     'nidorina', 'nidorino', 'ninetales', 'oddish', 'omanyte', 'omastar', 'onix', 'paras', 'parasect', 'persian', 'pidgeot', 'pidgeotto', 
     'pidgey', 'pikachu', 'pinsir', 'poliwag', 'poliwhirl', 'poliwrath', 'ponyta', 'porygon', 'primeape', 'psyduck', 'raichu', 'rapidash', 
     'raticate', 'rattata', 'rhydon', 'rhyhorn', 'sandshrew', 'sandslash', 'scyther', 'seadra', 'seaking', 'seel', 'shellder', 'slowbro', 
     'slowpoke', 'snorlax', 'spearow', 'squirtle', 'starmie', 'staryu', 'tangela', 'tauros', 'tentacool', 'tentacruel', 'vaporeon', 'venomoth', 
     'venonat', 'venusaur', 'victreebel', 'vileplume', 'voltorb', 'vulpix', 'wartortle', 'weedle', 'weepinbell', 'weezing', 'wigglytuff', 'zapdos', 'zubat']
    """
    similar_pokemon_types = [
        "nidoran-f", "nidoran-m", "nidorina", "nidorino", "nidoking", "nidoqueen", "rhyhorn", "rhydon", "sandslash", "kangaskhan"]
    df = df[df[LABEL_NAME].isin(similar_pokemon_types)]
    return df




CONTEXT = "Pokemon Object Classification based on images"
TARGET = CuratedTarget(raw_name=LABEL_NAME, task_type=SupervisedTask.MULTICLASS)
FEATURES = [CuratedFeature(raw_name=IMAGE_FEATURE_NAME, feat_type=FeatureType.IMAGE)]
IMAGE_FOLDER = ""
LOADING_FUNC = load_df
