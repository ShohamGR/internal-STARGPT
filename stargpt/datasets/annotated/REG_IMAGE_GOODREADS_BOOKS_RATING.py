import os
from os.path import join

from pandas import DataFrame

from tabstar2.utils.datasets import load_csv
from tabstar2.utils.images import download_url_image_column
from tabstar_paper.datasets.curation_objects import CuratedTarget, CuratedFeature
from tabstar_paper.datasets.objects import SupervisedTask, FeatureType

IMAGE_FEATURE_NAME = "img"

'''
Dataset Name: mdhamani/goodreads-books-100k/
====
Examples: 100000
====
URL: https://www.kaggle.com/mdhamani/goodreads-books-100k
====
Bag Of Tricks Paper:
goodreads: Predict the rating of a book based on its cover image and metadata including title, content
descriptions, genre and so on. This dataset originally stems from https://www.kaggle.com/
datasets/mdhamani/goodreads-books-100k. We remove non-English sentences, randomly
select 16% of the original dataset and split this subset at 3:1 ratio for new training and test sets. The
license of the original dataset: CC0 1.0.

Description: 

About Dataset
Context
This is a dataset was created as a personal project to learn scraping and to provide a usable and extensive dataset to the community. It lead me to one of my favorite books website goodreads. And then i just wrote the code for scraping (which BTW i'll upload soon on my github). And then it was just hours of scraping and at last this is the result.

Content
This dataset contains contains some of the generally required columns needed to express a book. If you have any questions or comments about this please feel free put them in discussions. I'll be happy to answer them 😄😄.

Acknowledgements
This dataset was gathered from goodreads, so a humble thanks to them.
Dataset image:- https://freebiesupply.com/logos/goodreads-logo/
====
Target Variable: rating (float64, 289 distinct): ['4.0', '0.0', '3.67', '3.75', '3.8', '3.88', '3.5', '3.86', '3.83', '3.94']
====
Features:

author (object, 68767 distinct): ['Mi-Ri Hwang', 'Willy Vandersteen', 'Yu-Rang Han', 'R.L. Stine', 'Elinor M. Brent-Dyer', 'Lynn Hagen', 'Anonymous', "Louis L'Amour", 'Lynne Graham', 'Agatha Christie']
bookformat (object, 202 distinct, 3.2% missing): ['Paperback', 'Hardcover', 'ebook', 'Kindle Edition', 'Mass Market Paperback', 'Unknown Binding', 'Nook', 'Audio CD', 'Board Book', 'Spiral-bound']
desc (object, 92499 distinct, 6.8% missing): ["This scarce antiquarian book is a facsimile reprint of the original. Due to its age, it may contain imperfections such as marks, notations, marginalia and flawed pages. Because we believe this work is culturally important, we have made it available as part of our commitment for protecting, preserving, and promoting the world's literature in affordable, high quality, modern editions that are true to the original work.", 'This book was converted from its physical edition to the digital format by a community of volunteers. You may find it for free on the web. Purchase of the Kindle edition includes wireless delivery.', 'This is a pre-1923 historical reproduction that was curated for quality. Quality assurance was conducted on each of these books in an attempt to remove books with imperfections introduced by the digitization process. Though we have made best efforts - the books may have occasional errors that do not impede the reading experience. We believe this work is culturally important and have elected to bring the book back into print as part of our continuing commitment to the preservation of printed works worldwide.', 'Many of the earliest books, particularly those dating back to the 1900s and before, are now extremely scarce and increasingly expensive. We are republishing these classic works in affordable, high quality, modern editions, using the original text and artwork.', "The story focuses on Kenichi, an average 16-year-old high school student who has been picked on his whole life. However, on his first day of class, he meets and befriends the mysterious transfer student, Miu FÅ«rinji. Driven by his desire to become stronger and to protect those around him, he follows her to RyÅ\x8dzanpaku, a dojo where those who are truly strong and have mastered their arts gather (RyÅ\x8dzanpaku comes from the Chinese story of Heroes of the Water Margin who train together at Liangshan æ¢\x81å±±). After learning basics from Miu, Kenichi is able to beat a high-ranking member of the school's karate club, and becomes a target for all the delinquents in the school. Kenichi's reason for training is to fulfill the promise he made to protect Miu. Subsequently, Kenichi's daily routine is divided between hellish training under the six masters of RyÅ\x8dzanpaku, and his fights against the members of Ragnarok, a gang that views him as either a possible ally or an impending threat to their plans.,After defeating Ragnarok, Kenichi faces a new enemy called Yomi, a group of disciples who are each personally trained by a master of an even bigger organization rivaling RyÅ\x8dzanpaku, Yami. While the masters of RyÅ\x8dzanpaku follow the principle of always sparing their opponents' lives (Katsujin-ken), the members of Yami believe that defeating an opponent is valid by any means including murder (Satsujin-ken). Caught in the struggle between the two factions, Kenichi, Miu and their ever growing team of allies join forces to fight the members of Yomi, while his masters confront the members of Yami in a battle to decide the future of the martial arts' world.", 'This is a reproduction of a book published before 1923. This book may have occasional imperfections such as missing or blurred pages, poor pictures, errant marks, etc. that were either part of the original artifact, or were introduced by the scanning process. We believe this work is culturally important, and despite the imperfections, have elected to bring it back into print as part of our continuing commitment to the preservation of printed works worldwide. We appreciate your understanding of the imperfections in the preservation process, and hope you enjoy this valuable book.', 'Important Note about PRINT ON DEMAND Editions: You are purchasing a print on demand edition of this book. This book is printed individually on uncoated (non-glossy) paper with the best quality printers available. The printing quality of this copy will vary from the original offset printing edition and may look more saturated. The information presented in this version is the same as the latest edition. Any pattern pullouts have been separated and presented as single pages. If the pullout patterns are missing, please contact c&t publishing.', 'This work has been selected by scholars as being culturally important, and is part of the knowledge base of civilization as we know it. This work was reproduced from the original artifact, and remains as true to the original work as possible. Therefore, you will see the original copyright references, library stamps (as most of these works have been housed in our most important libraries around the world), and other notations in the work. This work is in the public domain in the United States of America, and possibly other nations. Within the United States, you may freely copy and distribute this work, as no entity (individual or corporate) has a copyright on the body of the work.As a reproduction of a historical artifact, this work may contain missing or blurred pages, poor pictures, errant marks, etc. Scholars believe, and we concur, that this work is important enough to be preserved, reproduced, and made generally available to the public. We appreciate your support of the preservation process, and thank you for being an important part of keeping this knowledge alive and relevant.', '#NAME?', 'One in a series of twenty Old Testament verse-by-verse commentary books edited by Max Anders. Includes discussion starters, teaching plan, and more. Great for lay teachers and pastors alike.']
genre (object, 72129 distinct, 10.5% missing): ['Nonfiction', 'History', 'Games,Chess', 'Esoterica,Astrology', 'Poetry', 'Fiction', 'History,Nonfiction', 'Combat,Martial Arts', 'Music', 'Literature,Marathi']
img (object, 96955 distinct): ['', 'https___i.gr-assets.com_images_S_compressed.photo.goodreads.com_books_1358740654l_14548482.jpg', 'https___i.gr-assets.com_images_S_compressed.photo.goodreads.com_books_1183769366l_1454819.jpg', 'https___i.gr-assets.com_images_S_compressed.photo.goodreads.com_books_1311997975l_1454782.jpg', 'https___i.gr-assets.com_images_S_compressed.photo.goodreads.com_books_1358728563l_14547334.jpg', 'https___i.gr-assets.com_images_S_compressed.photo.goodreads.com_books_1347512706l_1454729.jpg', 'https___i.gr-assets.com_images_S_compressed.photo.goodreads.com_books_1344669994l_1454721.jpg', 'https___i.gr-assets.com_images_S_compressed.photo.goodreads.com_books_1388389122l_14546.jpg', 'https___i.gr-assets.com_images_S_compressed.photo.goodreads.com_books_1428080915l_1454686.jpg', 'https___i.gr-assets.com_images_S_compressed.photo.goodreads.com_books_1347278960l_14546758.jpg']
pages (int64, 1357 distinct): ['0', '192', '224', '32', '256', '288', '320', '304', '240', '128']
reviews (int64, 2950 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
title (object, 97588 distinct, 0.0% missing): ['Love in the Mask', 'Selected Poems', 'Cinderella', 'Coming Home', 'Redemption', 'Forbidden', 'Honggane', 'Inferno', 'Beauty and the Beast', 'Legacy']
totalratings (int64, 10536 distinct): ['1', '0', '2', '3', '4', '5', '6', '8', '7', '9']
'''

def load_df(dir_path: str) -> DataFrame:
    df = load_csv(dir_path, "GoodReads_100k_books.csv")
    img_folder = join(dir_path, IMAGE_FOLDER)
    df = download_url_image_column(df=df, img_folder=img_folder, img_col=IMAGE_FEATURE_NAME)
    df.drop(columns=['isbn', 'isbn13', 'link'], inplace=True)
    return df


CONTEXT = ""
TARGET = CuratedTarget(raw_name="rating", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name=IMAGE_FEATURE_NAME, feat_type=FeatureType.IMAGE)]
IMAGE_FOLDER = "downloaded_images"
LOADING_FUNC = load_df
