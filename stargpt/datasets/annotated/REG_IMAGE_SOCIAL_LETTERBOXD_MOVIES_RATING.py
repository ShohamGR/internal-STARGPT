from collections import defaultdict
from os.path import exists, join

from pandas import DataFrame, read_csv

from tabstar_paper.datasets.curation_objects import CuratedTarget, CuratedFeature
from tabstar_paper.datasets.objects import SupervisedTask, FeatureType


'''
Dataset Name: gsimonx37/letterboxd/
====
Examples: 12567
====
URL: https://www.kaggle.com/gsimonx37/letterboxd
====
Description: 
Data obtained using a program from the site letterboxd.com.

About letterboxd.com
"Letterboxd is a global social network for grass-roots film discussion and discovery. Use it as a diary to record and share your opinion about films as you watch them, or just to keep track of films you’ve seen in the past. Showcase your favorites on your profile page. Rate, review and tag films as you add them. Find and follow your friends to see what they’re enjoying. Keep a watchlist of films you’d like to see, and create lists/collections on any given topic. We’ve been described as "like GoodReads for movies". - from the site letterboxd.com

All film-related metadata used in Letterboxd, including actor, director and studio names, synopses, release dates, trailers and poster art is supplied by The Movie Database (TMDb)

Field descriptions:
The data contains the following fields:

movies.csv - basic information about films:
id - movie identifier (primary key);
name - the name of the film;
date - year of release of the film;
tagline - the slogan of the film;
description - description of the film;
minute - movie duration (in minutes);
rating - average rating of the film.
actors.csv - actors who took part in the filming of films:
id - movie identifier (foreign key);
name - name;
role - role.
crew.csv - film crew:
id - movie identifier (foreign key);
role - role in the film crew (director, screenwriter, etc.);
name - name.
languages.csv - in what languages ​​the films were shot:
id - movie identifier (foreign key);
type - type (primary, conversational, etc.)
language - film language.
studios.csv - film studios:
id - movie identifier (foreign key);
studio - film studio.
countries.csv - countries:
id - movie identifier (foreign key);
country - country.
genres.csv - film genres:
id - movie identifier (foreign key);
genre - film genre.
themes.csv - themes in films:
id - movie identifier (foreign key);
theme - the theme of the film.
releases.csv - movie releases:
id - movie identifier (foreign key);
country - release country;
date - release date of the film;
type - release type (theatrical, television, etc.) of the film.
rating - age rating of the film.
posters.csv - movie posters:
id - movie identifier (foreign key);
country - url address.
posters - movie posters.
Note:

tagline (film slogan) is present only in relatively modern films.
rating (average film rating) is missing for most films, since most likely there is not a sufficient number of estimates to calculate it.
If the value of type (movie language type) is Language, means the language is both primary and spoken (Primary language and Spoken language).
Some films do not have a poster (for natural reasons).
Found an error or inaccuracy in the data?
This dataset is the result of painstaking work. After collection and systematization, the data is checked for integrity and correctness. If you notice an error or inaccuracy in the data, or have a suggestion on how to improve the data set, please let me know.
====
Target Variable: rating (float64, 314 distinct): ['3.4', '3.41', '3.26', '3.34', '3.43', '3.46', '3.36', '3.47', '3.52', '3.49']
====
Features:

name (object, 12339 distinct): ['The Teacher', 'Mercy', 'The Line', 'Monster', 'First Love', 'Swan Song', 'Animal', 'The Visitor', 'Last Summer', 'Human Resources']
date (float64, 4 distinct): ['2022.0', '2023.0', '2021.0', '2024.0']
tagline (object, 4460 distinct): ['Face your demons.', 'The hunt is on.', 'Back for seconds.', 'Reap what you sow.', 'No one just disappears.', "It's only a matter of time.", 'Defy the odds', 'Find your voice.', 'Welcome to the family.', 'Time is running out.']
description (object, 12451 distinct): ['Away from school, during the winter holidays, three new stories take place while the Las Encinas students celebrate Christmas.', 'Choo Sang-woo is the epitome of an inflexible and strict rule-abiding person. Jang Jae-young is like a semantic error in the perfect world of Choo Sang-woo. Will Sang-woo be able to work with Jae-young as an artist and engineer?', "After going to extremes to cover up an accident, a corrupt cop's life spirals out of control when he starts receiving threats from a mysterious witness.", 'Jennifer and Meg Swift are two sisters who are very close despite living far apart. Jennifer is in Salt Lake City, running a successful restaurant she started with her late husband and raising her teenaged son Simon, Meg stayed in their hometown of Hazelwood, helping their parents run the local bakery.', 'Identical twins change their diets and lifestyles for eight weeks in a unique scientific experiment designed to explore how certain foods impact the body.', "Xbox almost didn't happen. Find out why in this behind-the-scenes, six-part series that takes you back to the scrappy beginnings of Microsoft's video game console. It's the untold story of the people behind the box, glitches and all.", 'After falling in love, a street-smart man and a wealthy woman from different worlds try to work out their differences.', 'Val Barber, a private investigator, is hired by a wealthy widow to find her missing granddaughter. Set in Dublin against the background of a global pandemic, Barber’s initial investigation into Sara’s disappearance quickly darkens. Secrets start surfacing in unexpected ways. Before too long, Barber finds himself entangled with powerful men of shady morals determined to thwart his investigation.  Has he bitten off more than he can chew?', "A short made by Ephraim Ryan as a one-man short for the 2021 MPC Short Film Festival following the theme of 'Everyday Life'. 'Between Days' won 1st place at the MPC Film Festival.", 'While Thomas and Oscar are very much in love, after their first foster child returns to his birth mother, they find that they have different ideas about what making a family actually means.']
minute (float64, 484 distinct): ['90.0', '84.0', '100.0', '95.0', '93.0', '96.0', '97.0', '91.0', '98.0', '85.0']
themes (object, 1757 distinct): ['', 'Epic heroes; Bollywood emotional dramas', 'Song and dance; Bollywood emotional dramas', 'Moving relationship stories; Powerful stories of heartbreak and suffering', 'Epic heroes; Superheroes in action-packed battles with villains', 'Moving relationship stories; Bollywood emotional dramas', 'Thrillers and murder mysteries; Suspenseful crime thrillers; Intriguing and suspenseful murder mysteries', 'Relationship comedy; Laugh-out-loud relationship entanglements', 'Moving relationship stories; Touching and sentimental family stories', 'Intense violence and sexual transgression; Twisted dark psychological thriller']
Language (object, 81 distinct): ['English', '', 'Spanish', 'French', 'Japanese', 'Korean', 'Portuguese', 'Tamil', 'Italian', 'Hindi']
Primary language (object, 86 distinct): ['', 'English', 'French', 'Spanish', 'German', 'Hindi', 'Korean', 'Italian', 'Chinese', 'Japanese']
Spoken language (object, 99 distinct): ['', 'Spanish', 'English', 'French', 'Portuguese', 'Italian', 'German', 'Russian', 'Japanese', 'Swedish']
genre_Animation (int64, 2 distinct): ['0', '1']
genre_Western (int64, 2 distinct): ['0', '1']
genre_War (int64, 2 distinct): ['0', '1']
genre_Comedy (int64, 2 distinct): ['0', '1']
genre Science Fiction (int64, 2 distinct): ['0', '1']
genre_Mystery (int64, 2 distinct): ['0', '1']
genre_Drama (int64, 2 distinct): ['0', '1']
genre_Action (int64, 2 distinct): ['0', '1']
genre_Crime (int64, 2 distinct): ['0', '1']
genre_Adventure (int64, 2 distinct): ['0', '1']
genre_Romance (int64, 2 distinct): ['0', '1']
genre_Thriller (int64, 2 distinct): ['0', '1']
genre_Fantasy (int64, 2 distinct): ['0', '1']
genre TV Movie (int64, 2 distinct): ['0', '1']
genre_Music (int64, 2 distinct): ['0', '1']
genre_History (int64, 2 distinct): ['0', '1']
genre_Horror (int64, 2 distinct): ['0', '1']
genre_Family (int64, 2 distinct): ['0', '1']
genre_Documentary (int64, 2 distinct): ['0', '1']
Poster (object, 12567 distinct): ['1000001.jpg', '1000003.jpg', '1000006.jpg', '1000009.jpg', '1000013.jpg', '1000019.jpg', '1000020.jpg', '1000021.jpg', '1000027.jpg', '1000028.jpg']
'''

LABEL_NAME = "rating"
IMAGE_FEATURE_NAME = "Poster"


def load_df(dir_path: str) -> DataFrame:
    '''We should be careful of leakage with other datasets.
    REG_SOCIAL_MOVIES_ROTTEN_TOMATOES: Movies up to 2019.
    BIN_SOCIAL_IMDB_GENRE_PREDICTION: Movies up to 2016.
    REG_SOCIAL_MOVIES_DATASET_REVENUE: Movies up to 2017.
    REG_SOCIAL_MOVIES_DATASET_REVENUE: Movies up to 2020.

    So, we can take easily 2021 and forward'''
    movies_path = join(dir_path, "movies.csv")
    df = read_csv(movies_path)
    df = df[df['date'] >= 2021]
    df = df[df[LABEL_NAME].notnull()]
    df = add_themes(df, dir_path=dir_path)
    df = add_language(df, dir_path=dir_path)
    df = add_genres(df, dir_path=dir_path)
    df[IMAGE_FEATURE_NAME] = df['id'].apply(lambda i: validate_image(i, dir_path=dir_path))
    df.drop(columns=['id'], inplace=True)
    return df

def validate_image(img_id: str, dir_path: str) -> str | None:
    img = f"{img_id}.jpg"
    img_path = join(dir_path, IMAGE_FOLDER, img)
    if not exists(img_path):
        return None
    return img_path

def add_themes(df: DataFrame, dir_path: str) -> DataFrame:
    themes = read_csv(join(dir_path, "themes.csv"))
    # Themes is duplicate and has many entries. concatenate everything
    id2themes = defaultdict(list)
    for _, row in themes.iterrows():
        id2themes[row['id']].append(row['theme'])
    id2theme = {movie_id: "; ".join(theme_list) for movie_id, theme_list in id2themes.items()}
    df['themes'] = df['id'].apply(lambda i: id2theme.get(i, ""))
    return df

def add_language(df: DataFrame, dir_path: str) -> DataFrame:
    languages = read_csv(join(dir_path, "languages.csv"))
    for col in ['Language', 'Primary language', 'Spoken language']:
        lang_df = languages.copy()
        lang_df = lang_df[lang_df['type'] == col]
        id2lang = {row['id']: row['language'] for _, row in lang_df.iterrows()}
        df[col] = df['id'].apply(lambda i: id2lang.get(i, ""))
    return df


def add_genres(df: DataFrame, dir_path: str) -> DataFrame:
    genres = read_csv(join(dir_path, "genres.csv"))
    all_genres = set(genres['genre'].unique().tolist())
    for genre in all_genres:
        genre_df = genres[genres['genre'] == genre]
        id_set = set(genre_df['id'].tolist())
        df[f"genre_{genre}"] = df['id'].apply(lambda i: 1 if i in id_set else 0)
    return df

CONTEXT = "Movies Posters Rating"
TARGET = CuratedTarget(raw_name=LABEL_NAME, task_type=SupervisedTask.REGRESSION)
TEXT_FEATURES = [CuratedFeature(raw_name=tf, feat_type=FeatureType.TEXT) for tf in ["name", "tagline", "description", "themes"]]
FEATURES = [CuratedFeature(raw_name=IMAGE_FEATURE_NAME, feat_type=FeatureType.IMAGE)] + TEXT_FEATURES
IMAGE_FOLDER = "posters"
LOADING_FUNC = load_df
