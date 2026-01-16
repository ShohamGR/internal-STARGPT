from os.path import exists, join

from pandas import DataFrame

from tabstar2.utils.datasets import load_csv
from tabstar_paper.datasets.curation_objects import CuratedTarget, CuratedFeature
from tabstar_paper.datasets.objects import SupervisedTask, FeatureType


'''
Dataset Name: sqdartemy/minecraft-screenshots-dataset-with-features/
====
Examples: 6089
====
URL: https://www.kaggle.com/sqdartemy/minecraft-screenshots-dataset-with-features
====
Description: 
Minecraft Screenshots Dataset with Features
About 6k original Minecraft Gameplay Screenshots along with .csv features file.


About Dataset
Overview
This dataset consists of Minecraft gameplay screenshots accompanied by a detailed .csv file containing metadata and decision-making annotations for each image. The dataset captures various in-game scenarios, player states, and the corresponding decisions made based on specific rules. It is designed to facilitate research and development in areas such as game AI, player behavior analysis, decision-making processes, and machine learning applications within gaming contexts.

P.S. This dataset was used for this paper: https://doi.org/10.36227/techrxiv.174002528.84035428/v1

Dataset Components
1. Screenshots
High-quality images capturing diverse in-game scenarios and player activities.
Each screenshot is uniquely titled and corresponds to a specific entry in the metadata file.
2. Metadata (features_and_decision.csv)
A structured file containing detailed information about each screenshot.
The metadata includes both the observed game state variables and the decisions made according to predefined rules.
Data Fields and Annotations
Input Variables
screenshot_title: The filename of the corresponding screenshot.
activity: The player's current activity.
Possible values (with priority in parentheses): archery(2), building(4), fighting(1), mining(3), swimming(5), walking(6).
hearts: Player's health level, ranging from 0 to 20.
light_lvl: The ambient light level in the game environment.
Possible values: high, mid, low.
in_hand_item: The item the player is holding.
Possible values: pickaxe, sword, axe, bow, crossbow, block, miscellaneous.
target_mob: The type of mob the player is interacting with.
Possible values: zombie, spider, skeleton, creeper, ender, other.
Decision Variables
The decision fields are derived from the input variables based on specific rules:

decision_activity: Decisions related to the player's activity.
Rules:

archery → give_resistance
building → give_jump_boost
fighting → give_strength
mining → give_haste
swimming → give_water_breathing
walking → give_speed
decision_hearts: Decisions concerning the player's health.
Rules:

15 ≤ hearts < 20 → give_regeneration_1
10 ≤ hearts < 15 → give_regeneration_2
5 ≤ hearts < 10 → give_regeneration_3
0 ≤ hearts < 5 → give_regeneration_4
decision_light: Actions taken regarding light levels.
Rules:

high → no_decision_for_light
mid → place_light_source
low → place_light_source
decision_mob: Decisions involving mobs.
Rules:

creeper → go_back
skeleton → take_bow (if not already in hand)
zombie → take_sword (if not already in hand or if an axe is not in hand)
spider → take_sword (if not already in hand or if an axe is not in hand)
other → no_decision_for_mob
Decision Rules Explanation
Activity-Based Decisions (decision_activity)
Each player activity triggers a specific beneficial effect:

Archery: Player receives give_resistance to enhance defense.
Building: Player receives give_jump_boost to aid in construction.
Fighting: Player receives give_strength to increase attack power.
Mining: Player receives give_haste to speed up mining.
Swimming: Player receives give_water_breathing to stay underwater longer.
Walking: Player receives give_speed to move faster.
Health-Based Decisions (decision_hearts)
The player's health determines the level of regeneration effect:

15 to 19 Hearts: give_regeneration_1
10 to 14 Hearts: give_regeneration_2
5 to 9 Hearts: give_regeneration_3
0 to 4 Hearts: give_regeneration_4
Light Level Decisions (decision_light)
Ambient light levels dictate whether to place a light source:

High Light Level: no_decision_for_light
Mid Light Level: place_light_source
Low Light Level: place_light_source
Mob Interaction Decisions (decision_mob)
The type of mob influences the player's action:

Creeper: Player should go_back to avoid explosion.
Skeleton: Player should take_bow if not already equipped.
Zombie/Spider: Player should take_sword if not already equipped or if an axe is not in hand.
Other Mobs: no_decision_for_mob
Potential Applications
Game AI Development: Enhance non-player character (NPC) behaviors by analyzing player decision patterns.
Machine Learning Models: Train models to predict player actions based on environmental and state variables.
Behavioral Studies: Examine how in-game factors like health and light levels influence player choices.
Educational Tools: Utilize the dataset for teaching data analysis, AI, and decision-making processes in gaming.
Data Access and Format
Format: The dataset is provided in a compressed folder containing all screenshots and the features_and_decision.csv file.
Access: Available for download at [provide download link or repository information].
Data Integrity: Ensure that the filenames in the screenshot_title field match the actual screenshot files for accurate data mapping.
Licensing and Usage
License: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
Attribution: When using this dataset, please cite it as follows:
[Your Name], "Minecraft Gameplay Decision Dataset with Screenshots," [Year]. Available at [URL or DOI if applicable].
Attribution to Mojang Studios:
Minecraft® is a trademark of Mojang Studios. This dataset includes in-game screenshots used under Mojang's Brand and Asset Usage Guidelines for non-commercial, educational purposes.

Usage Restrictions:
The dataset is intended for non-commercial, research, and educational purposes only.
Commercial use is prohibited without explicit permission from Mojang Studios.

Notes
The dataset aims to preserve the authenticity of gameplay while providing structured data for analysis.
All categorical fields in the metadata are standardized to maintain consistency across entries.
The decision rules are explicitly defined to enable users to understand the logic behind each decision annotation.
Users are encouraged to share insights or improvements derived from the dataset to contribute to the gaming and AI community.
Disclaimer
This dataset is intended for research and educational purposes. Minecraft® and all related assets are the property of Mojang Studios and Microsoft. This dataset is not affiliated with or endorsed by Mojang Studios or Microsoft.

Summary
This dataset provides a comprehensive resource for analyzing decision-making processes in Minecraft gameplay. By incorporating explicit rules that map game state variables to decisions, the dataset offers valuable insights for developing AI models, studying player behavior, and advancing research in gaming and artificial intelligence.

Next Steps
Download the Dataset: Access the dataset from the provided link and explore the screenshots and metadata.
Analyze the Decision Rules: Utilize the explicit decision rules to understand how in-game states influence player decisions.
Develop Applications: Apply the dataset to your specific area of interest, such as training machine learning models or conducting behavioral analysis.
Share Your Findings: Contribute back to the community by sharing any insights, models, or applications developed using this dataset.
Acknowledgments
Thank you for your interest in this dataset. Your engagement helps foster a collaborative environment for advancing research and development in gaming and AI.
====
Target Variable: target_mob (object, 7 distinct): ['no_mob', 'other', 'skeleton', 'zombie', 'spider', 'ender', 'creeper']
====
Features:

screenshot_title (object, 5574 distinct): ['mining (293).png', 'mining (296).png', 'mining (304).png', 'mining (317).png', 'mining (321).png', 'mining (322).png', 'mining (323).png', 'mining (325).png', 'mining (328).png', 'mining (329).png']
activity (object, 6 distinct): ['archery', 'fighting', 'swimming', 'walking', 'mining', 'building']
hearts (int64, 21 distinct): ['20', '4', '7', '6', '15', '11', '14', '18', '5', '12']
light_lvl (object, 3 distinct): ['high', 'mid', 'low']
in_hand_item (object, 8 distinct): ['bow', 'no_item', 'sword', 'pickaxe', 'miscellaneous', 'block', 'axe', 'crossbow']
decision_activity (object, 6 distinct): ['give_resistance', 'give_strength', 'give_water_breathing', 'give_speed', 'give_haste', 'give_jump_boost']
decision_hearts (object, 5 distinct): ['no_decision_for_hearts', 'give_regeneration_3', 'give_regeneration_4', 'give_regeneration_2', 'give_regeneration_1']
decision_light (object, 3 distinct): ['no_decision_for_light', 'place_light_source', 'palce_light_source']
decision_mob (object, 4 distinct): ['no_decision_for_mob', 'take_sword', 'take_bow', 'go_back']
'''

SCREENSHOT = "screenshot_title"

def load_df(dir_path: str) -> DataFrame:
    df = load_csv(dir_path, "features_and_decisions.csv")
    _validate_img(df, dir_path=dir_path)
    return df

def _validate_img(df: DataFrame, dir_path: str):
    for img in df[SCREENSHOT]:
        img_path = join(dir_path, IMAGE_FOLDER, img)
        if not exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

CONTEXT = ""
TARGET = CuratedTarget(raw_name='target_mob', task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name=SCREENSHOT, feat_type=FeatureType.IMAGE)]
IMAGE_FOLDER = "screenshots/screenshots"
LOADING_FUNC = load_df
