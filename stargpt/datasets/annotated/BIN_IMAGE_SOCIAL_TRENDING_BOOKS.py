from os.path import join

from pandas import DataFrame, read_csv

from tabstar2.utils.images import download_url_image_column
from tabstar_paper.datasets.curation_objects import CuratedTarget, CuratedFeature
from tabstar_paper.datasets.objects import SupervisedTask, FeatureType


'''
Dataset Name: kuchhbhi/treding-book-dataset/
====
Examples: 1000
====
URL: https://www.kaggle.com/kuchhbhi/treding-book-dataset
====
Description: 
About Dataset
📚 Books Data 💡
📖 Dataset Description
This dataset, lovingly curated by our team, serves as a valuable resource for exploring the world of literature. It provides insights into various aspects of books and is designed for data cleaning, transformation, analysis, and visualization.

Context:
The dataset, created using Python libraries like Requests, BeautifulSoup (bs4), and Pandas, features information on a diverse collection of books. It was carefully gathered from reputable online bookstores, libraries, and literary websites.

Inspiration:
Our motivation to compile this dataset is to facilitate data-driven research and analysis in the field of literature and book publishing.

Data Source: The book data in this dataset was scraped from the Books to Scrape website. This source provides a diverse collection of books from various genres, making it an excellent resource for data analysis and research within the literary domain.

Data Columns (Total 12 Columns):

Title: The title of the book, representing the book's name.
Category: The category or genre to which each book belongs.
Image: URLs or references to images associated with the books.
Rating: The rating or review score of the book.
Description: A brief description or summary of the book's content. (Note: There are 998 non-null entries, implying two missing descriptions.)
UPC (Universal Product Code): A unique product identifier for each book.
Product Type: The type or format of the book, such as hardcover or paperback.
Price (excl. tax): The price of the book without taxes.
Price (incl. tax): The price of the book, including taxes.
Tax: The tax amount associated with the book.
Availability: Information about the book's availability for purchase.
Number of Reviews: The number of reviews or ratings provided by readers for each book.
This dataset provides a comprehensive view of books, encompassing their titles, genres, ratings, descriptions, pricing, and availability. It is well-suited for data analysis, visualization, and research within the literary and book retail domain.

📊 Use Cases
Data Cleaning: Practice and demonstrate data cleaning techniques, such as handling missing values and standardizing data.
Data Transformation: Categorize books by genre, author, or publication date.
Analysis: Explore trends in book publishing, popular genres, and authorship over time.
Visualization: Create visualizations of book trends and insights through plots, graphs, and charts.
Please note that while this dataset was created with great care, it is essential to respect copyright and data usage policies when working with book information. The dataset provided here is intended for educational and research purposes.
====
Target Variable: Availability (bool, 2 distinct): ['1', '0']
====
Features:

Title (object, 999 distinct): ['The Star-Touched Queen', 'The Bone Hunters (Lexy Vaughan & Steven Macaulay #2)', 'The Beast (Black Dagger Brotherhood #14)', 'Some Women', 'Shopaholic Ties the Knot (Shopaholic #3)', 'Paper and Fire (The Great Library #2)', 'Outlander (Outlander #1)', 'Orchestra of Exiles: The Story of Bronislaw Huberman, the Israel Philharmonic, and the One Thousand Jews He Saved from Nazi Horrors', 'No One Here Gets Out Alive', 'Night Shift (Night Shift #1-20)']
Category (object, 50 distinct): ['Default', 'Nonfiction', 'Sequential Art', 'Add a comment', 'Fiction', 'Young Adult', 'Fantasy', 'Romance', 'Mystery', 'Food and Drink']
Book Cover (object, 1000 distinct): ['https___books.toscrape.com_media_cache_fe_72_fe72f0532301ec28892ae79a629a293c.jpg', 'https___books.toscrape.com_media_cache_08_e9_08e94f3731d7d6b760dfbfbc02ca5c62.jpg', 'https___books.toscrape.com_media_cache_ee_cf_eecfe998905e455df12064dba399c075.jpg', 'https___books.toscrape.com_media_cache_c0_59_c05972805aa7201171b8fc71a5b00292.jpg', 'https___books.toscrape.com_media_cache_ce_5f_ce5f052c65cc963cf4422be096e915c9.jpg', 'https___books.toscrape.com_media_cache_6b_07_6b07b77236b7c80f42bd90bf325e69f6.jpg', 'https___books.toscrape.com_media_cache_e1_1b_e11bea016d0ae1d7e2dd46fb3cb870b7.jpg', 'https___books.toscrape.com_media_cache_97_36_9736132a43b8e6e3989932218ef309ed.jpg', 'https___books.toscrape.com_media_cache_d1_2d_d12d26739b5369a6b5b3024e4d08f907.jpg', 'https___books.toscrape.com_media_cache_d1_7a_d17a3e313e52e1be5651719e4fba1d16.jpg']
Rating (object, 5 distinct): ['One', 'Three', 'Five', 'Two', 'Four']
Description (object, 999 distinct): ['Unknown Value', 'On her sixteenth birthday, orphan Himari Momochi inherits her ancestral estate that sheâ\x80\x99s never seen. Momochi House exists on the barrier between the human and spiritual realms, and Himari is meant to act as guardian between the two worlds. But on the day she moves in, she finds three handsome squatters already living in the house, and one seems to have already taken over On her sixteenth birthday, orphan Himari Momochi inherits her ancestral estate that sheâ\x80\x99s never seen. Momochi House exists on the barrier between the human and spiritual realms, and Himari is meant to act as guardian between the two worlds. But on the day she moves in, she finds three handsome squatters already living in the house, and one seems to have already taken over her role! ...more', 'The award-winning author of Valhalla brings back archaeologist Lexy Vaughan and retired Air Force officer Steve Macaulay, as they race to save a priceless discovery from disappearing forever.â\x80¦ One of the greatest archaeological finds of all time, Peking Man, the 780,000-year-old remains of our earliest known human ancestor, disappeared during World War II from a cargo shi The award-winning author of Valhalla brings back archaeologist Lexy Vaughan and retired Air Force officer Steve Macaulay, as they race to save a priceless discovery from disappearing forever.â\x80¦ Â\xa0 One of the greatest archaeological finds of all time, Peking Man, the 780,000-year-old remains of our earliest known human ancestor, disappeared during World War II from a cargo ship bound for America. Â\xa0 Now the Chinese government is fighting to keep a new religion from taking holdâ\x80\x94a faith based on the belief that Peking Man is God. And they dispatch ruthless operatives to find and destroy the worldâ\x80\x99s most priceless fossil. Â\xa0 But the U.S. government has its own team on the hunt. From the mountains of Bavaria to the jungles of Central America and across the vast Pacific, Professor Barnaby Finchem, his brilliant protÃ©gÃ©, Lexy Vaughan, and pilot Steve Macaulay will brave the wrath of nature and of man to win a race against unbridled tyranny.â\x80¦ ...more', 'Rhage and Mary return in a new novel of the Black Dagger Brotherhood, a series â\x80\x9cso popular, I donâ\x80\x99t think thereâ\x80\x99s a reader today who hasnâ\x80\x99t at least heard of [it]â\x80\x9d (USA Today).Nothing is as it used to be for the Black Dagger Brotherhood. After avoiding war with the Shadows, alliances have shifted and lines have been drawn. The slayers of the Lessening Society are stronger Rhage and Mary return in a new novel of the Black Dagger Brotherhood, a series â\x80\x9cso popular, I donâ\x80\x99t think thereâ\x80\x99s a reader today who hasnâ\x80\x99t at least heard of [it]â\x80\x9d (USA Today).Nothing is as it used to be for the Black Dagger Brotherhood. After avoiding war with the Shadows, alliances have shifted and lines have been drawn. The slayers of the Lessening Society are stronger than ever, preying on human weakness to acquire more money, more weapons, more power. But as the Brotherhood readies for an all-out attack on them, one of their own fights a battle within himselfâ\x80¦For Rhage, the Brother with the biggest appetites, but also the biggest heart, life was supposed to be perfectâ\x80\x94or at the very least, perfectly enjoyable. Mary, his beloved shellan, is by his side and his King and his brothers are thriving. But Rhage canâ\x80\x99t understandâ\x80\x94or controlâ\x80\x94the panic and insecurity that plague himâ\x80¦And that terrifies himâ\x80\x94as well as distances him from his mate. After suffering mortal injury in battle, Rhage must reassess his prioritiesâ\x80\x94and the answer, when it comes to him, rocks his world...and Maryâ\x80\x99s. But Mary is on a journey of her own, one that will either bring them closer together or cause a split that neither will recover from... ...more', 'An engrossing and thought provoking novel that examines the intricacies of marriage, friendship, and the power of unexpected connectionsâ\x80¦Annabel Ford has everything under control, devoting her time to her twin five-year-old boys and to keeping her household running seamlessly. So when her husband of a decade announces that heâ\x80\x99s leaving her, without warning, sheâ\x80\x99s blindside An engrossing and thought provoking novel that examines the intricacies of marriage, friendship, and the power of unexpected connectionsâ\x80¦Annabel Ford has everything under control, devoting her time to her twin five-year-old boys and to keeping her household running seamlessly. So when her husband of a decade announces that heâ\x80\x99s leaving her, without warning, sheâ\x80\x99s blindsided. And suddenly her world begins to unravel.Single mother Piper Whitley has always done her best to balance it allâ\x80\x94raising her daughter Fern by herself and advancing her career as a crime reporter. Only now that sheâ\x80\x99s finally met the man of her dreams, Fernâ\x80\x99s absentee father arrives on the scene and throws everything into a tailspin.Married to the heir of a thriving media conglomerate, Mackenzie Mead has many reasons to count her blessings. But with an imperious mother-in-lawâ\x80\x94whoâ\x80\x99s also her bossâ\x80\x94and a husband with whom she can no longer seem to connect, something has to give.On the surface, these three women may not have much in common. Yet when their lives are thrust together and unlikely friendships are formedâ\x80\x94at a time when they all need someone to lean onâ\x80\x94Annabel, Piper, and Mackenzie band together to help each navigate their new realities. ...more', 'Life has been good for Becky Bloomwood: Sheâ\x80\x99s become the best personal shopper at Barneys, she and her successful entrepreneurial boyfriend, Luke, are living happily in Manhattanâ\x80\x99s West Village, and her new next-door neighbor is a fashion designer! But with her best friend, Suze, engaged, how can Becky fail to notice that her own ring finger is bare? Not that sheâ\x80\x99s been th Life has been good for Becky Bloomwood: Sheâ\x80\x99s become the best personal shopper at Barneys, she and her successful entrepreneurial boyfriend, Luke, are living happily in Manhattanâ\x80\x99s West Village, and her new next-door neighbor is a fashion designer! But with her best friend, Suze, engaged, how can Becky fail to notice that her own ring finger is bare? Not that sheâ\x80\x99s been thinking of marriage (or diamonds) or anything . . . Then Luke proposes! Bridal registries dance in Beckyâ\x80\x99s head. Problem is, two other people are planning her wedding: Beckyâ\x80\x99s overjoyed mother has been waiting forever to host a backyard wedding, with the bride resplendent in Mumâ\x80\x99s frilly old gown. While Lukeâ\x80\x99s high-society mother is insisting on a glamorous, all-expenses-paid affair at the Plaza. Both weddings for the same day. And Becky canâ\x80\x99t seem to turn down either one. Can everyoneâ\x80\x99s favorite shopaholic tie the knot before everything unravels? ...more', 'In Ink and Bone, New York Times bestselling author Rachel Caine introduced a world where knowledge is power, and power corrupts absolutely. Now, she continues the story of those who dare to defy the Great Libraryâ\x80\x94and rewrite historyâ\x80¦With an iron fist, The Great Library controls the knowledge of the world, ruthlessly stamping out all rebellion, forbidding the personal owner In Ink and Bone, New York Times bestselling author Rachel Caine introduced a world where knowledge is power, and power corrupts absolutely. Now, she continues the story of those who dare to defy the Great Libraryâ\x80\x94and rewrite historyâ\x80¦With an iron fist, The Great Library controls the knowledge of the world, ruthlessly stamping out all rebellion, forbidding the personal ownership of books in the name of the greater good.Jess Brightwell has survived his introduction to the sinister, seductive world of the Library, but serving in its army is nothing like he envisioned. His life and the lives of those he cares for have been altered forever. His best friend is lost, and Morgan, the girl he loves, is locked away in the Iron Tower and doomed to a life apart.Embarking on a mission to save one of their own, Jess and his band of allies make one wrong move and suddenly find themselves hunted by the Libraryâ\x80\x99s deadly automata and forced to flee Alexandria, all the way to London.But Jessâ\x80\x99s home isnâ\x80\x99t safe anymore. The Welsh army is coming, London is burning, and soon, Jess must choose between his friends, his family, or the Library willing to sacrifice anything and anyone in the search for ultimate controlâ\x80¦ ...more', 'The year is 1945. Claire Randall, a former combat nurse, is just back from the war and reunited with her husband on a second honeymoon when she walks through a standing stone in one of the ancient circles that dot the British Isles. Suddenly she is a Sassenachâ\x80\x94an â\x80\x9coutlanderâ\x80\x9dâ\x80\x94in a Scotland torn by war and raiding border clans in the year of Our Lord...1743.Hurled back in ti The year is 1945. Claire Randall, a former combat nurse, is just back from the war and reunited with her husband on a second honeymoon when she walks through a standing stone in one of the ancient circles that dot the British Isles. Suddenly she is a Sassenachâ\x80\x94an â\x80\x9coutlanderâ\x80\x9dâ\x80\x94in a Scotland torn by war and raiding border clans in the year of Our Lord...1743.Hurled back in time by forces she cannot understand, Claire is catapulted into the intrigues of lairds and spies that may threaten her life, and shatter her heart. For here James Fraser, a gallant young Scots warrior, shows her a love so absolute that Claire becomes a woman torn between fidelity and desireâ\x80\x94and between two vastly different men in two irreconcilable lives. ...more', 'The compelling biography of the violinist who founded the Palestine Symphony Orchestra and saved hundreds of people from Hitlerâ\x80\x94as seen in Josh Aronsonâ\x80\x99s documentary Orchestra of Exiles.â\x80\x9cThe true artist does not create art as an end in itself. He creates art for human beings. Humanity is the goal.â\x80\x9dâ\x80\x94Bronislaw HubermanAt fourteen, Bronislaw Huberman played the Brahms Violin The compelling biography of the violinist who founded the Palestine Symphony Orchestra and saved hundreds of people from Hitlerâ\x80\x94as seen in Josh Aronsonâ\x80\x99s documentary Orchestra of Exiles.â\x80\x9cThe true artist does not create art as an end in itself. He creates art for human beings. Humanity is the goal.â\x80\x9dâ\x80\x94Bronislaw HubermanAt fourteen, Bronislaw Huberman played the Brahms Violin Concerto in Viennaâ\x80\x94 winning high praise from the composer himself, who was there. Instantly famous, Huberman began touring all over the world and received invitations to play for royalty across Europe. But after witnessing the tragedy of World War I, he committed his phenomenal talent and celebrity to aid humanity.After studying at the Sorbonne in Paris, Huberman joined the ranks of Sigmund Freud and Albert Einstein in calling for peace through the Pan European Movement. But when hope for their noble vision was destroyed by the rise of Nazism, Huberman began a crusade that would become his greatest legacyâ\x80\x94the creation, in 1936, of the Palestine Symphony, which twelve years later became the Israel Philharmonic Orchestra.In creating this world-level orchestra, Huberman miraculously arranged for the very best Jewish musicians and their families to emigrate from Nazi-threatened territories. His tireless campaigning for the projectâ\x80\x94including a marathon fundraising concert tour across Americaâ\x80\x94ultimately saved nearly a thousand Jews from the approaching Holocaust. Inviting the great Arturo Toscanini to conduct the orchestraâ\x80\x99s first concert, Hubermanâ\x80\x99s clarion call of art over cruelty was heard around the world. His story contains estraordinary adventures, riches and royalty, politicians and broken promises, losses and triumphs. Against near impossible obstacles, Huberman refused to give up on his dream to create a unique and life-saving orchestra of exiles which was one of the great cultural achievements of the 20th century.Includes Photographs ...more', 'Here is Jim Morrison in all his complexity-singer, philosopher, poet, delinquent-the brilliant, charismatic, and obsessed seeker who rejected authority in any form, the explorer who probed "the bounds of reality to see what would happen..." Seven years in the writing, this definitive biography is the work of two men whose empathy and experience with Jim Morrison uniquely p Here is Jim Morrison in all his complexity-singer, philosopher, poet, delinquent-the brilliant, charismatic, and obsessed seeker who rejected authority in any form, the explorer who probed "the bounds of reality to see what would happen..." Seven years in the writing, this definitive biography is the work of two men whose empathy and experience with Jim Morrison uniquely prepared them to recount this modern tragedy: Jerry Hopkins, whose famous Presley biography, Elvis, was inspired by Morrison\'s suggestion, and Danny Sugerman, confidant of and aide to the Doors. With an afterword by Michael McClure. ...more']
Number of reviews (int64, 1 distinct): ['0']
Price in £ with tax (float64, 903 distinct): ['27.88', '16.28', '39.24', '44.18', '37.34', '59.45', '51.32', '38.77', '40.83', '33.14']
'''

IS_AVAILABLE = "Availability"
IMAGE_FEATURE_NAME = "Book Cover"


def load_df(dir_path: str) -> DataFrame:
    csv_path = join(dir_path, "book_data.csv")
    df = read_csv(csv_path)
    img_folder = join(dir_path, IMAGE_FOLDER)
    df = download_url_image_column(df=df, img_folder=img_folder, img_col="Image")
    df.rename(columns={"Image": IMAGE_FEATURE_NAME}, inplace=True)
    df[IS_AVAILABLE] = df[IS_AVAILABLE].apply(parse_availability)
    df['Price in £ with tax'] = df['Price (incl. tax)'].apply(parse_price)
    df.drop(columns=['Product Type', 'UPC', 'Price (incl. tax)', 'Price (excl. tax)', 'Tax'],
            inplace=True)
    return df


def parse_availability(availability_str: str) -> bool:
    # Example input: "In stock (22 available)"
    prefix = "In stock ("
    assert availability_str.startswith(prefix)
    availability_str = availability_str[len(prefix):]
    suffix = " available)"
    assert availability_str.endswith(suffix)
    availability_str = availability_str[:-len(suffix)]
    availability = int(availability_str)
    return availability > 5

def parse_price(price_str: str) -> float:
    # Example input: "Â£53.74"
    price_str = price_str.replace('Â£', '')
    return float(price_str)

CONTEXT = ""
TARGET = CuratedTarget(raw_name=IS_AVAILABLE, task_type=SupervisedTask.BINARY)
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name=IMAGE_FEATURE_NAME, feat_type=FeatureType.IMAGE),
            CuratedFeature(raw_name="Title", feat_type=FeatureType.TEXT),
            CuratedFeature(raw_name="Description", feat_type=FeatureType.TEXT),
]
IMAGE_FOLDER = "downloaded_images"
LOADING_FUNC = load_df
