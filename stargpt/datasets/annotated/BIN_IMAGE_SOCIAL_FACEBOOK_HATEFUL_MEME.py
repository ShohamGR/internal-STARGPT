from os.path import join

from pandas import DataFrame

from tabstar_paper.datasets.curation_objects import CuratedTarget, CuratedFeature
from tabstar_paper.datasets.objects import SupervisedTask, FeatureType
from tabstar_paper.utils.io_handlers import load_json_lines


'''
Dataset Name: parthplc/facebook-hateful-meme-dataset/
====
Examples: 9000
====
URL: https://www.kaggle.com/parthplc/facebook-hateful-meme-dataset
====
Description:
The Hateful Memes Challenge README
The Hateful Memes Challenge is a dataset and benchmark created by Facebook AI to drive and measure progress on multimodal reasoning and understanding. The task focuses on detecting hate speech in multimodal memes.

Please see the paper for further details:

[The Hateful Memes Challenge: Detecting Hate Speech in Multimodal Memes
D. Kiela, H. Firooz, A. Mohan, V. Goswami, A. Singh, P. Ringshia, D. Testuggine](
https://arxiv.org/abs/2005.04790)

Dataset details
The files for this folder are arranged as follows:

img/ - the PNG images
train.jsonl - the training set
dev.jsonl - the development set
test.jsonl - the “seen” test set

An additional “unseen” test set will be released at a later date under the NeurIPS 2020 competition. Please see https://ai.facebook.com/hatefulmemes. The competition rules are provided on the competition website.

The .jsonl format contains one JSON-encoded example per line, each of which has the following fields:

‘text’ - the text occurring in the meme
‘img’ - the path to the image in the img/ directory
‘label’ - the label for the meme (0=not-hateful, 1=hateful), provided for train and dev

The metric to use is AUROC. You may also report accuracy in addition, since this is more interpretable. To compute these metrics, we recommend the roc_auc_score and accuracy_score methods in sklearn.metrics, with default settings.

Note on Annotator Accuracy
As is to be expected with a dataset of this size and nature, some of the examples in the training set have been misclassified. We are not claiming that our dataset labels are completely accurate, or even that all annotators would agree on a particular label. Misclassifications, although possible, should be very rare in the dev and seen test set, however, and we will take extra care with the unseen test set.

As a reminder, the annotations collected for this dataset were not collected using Facebook annotators and we did not employ Facebook’s hate speech policy. As such, the dataset labels do not in any way reflect Facebook’s official stance on this matter.

License
The dataset is licensed under the terms in the LICENSE.txt file.

Image Attribution
If you wish to display example memes in your paper, please provide the following attribution:

Image is a compilation of assets, including ©Getty Image.

Citations
If you wish to cite this work, please use the following BiBTeX:

@inproceedings{Kiela2020TheHM,
  title={The Hateful Memes Challenge: Detecting Hate Speech in Multimodal Memes},
  author={Douwe Kiela and Hamed Firooz and Aravind Mohan and Vedanuj Goswami and Amanpreet Singh and Pratik Ringshia and Davide Testuggine},
  year={2020}
}
====
Target Variable: label (float64, 2 distinct): ['0.0', '1.0']
====
Features:

img (object, 9000 distinct): ['img/42953.png', 'img/23058.png', 'img/13894.png', 'img/37408.png', 'img/82403.png', 'img/16952.png', 'img/76932.png', 'img/70914.png', 'img/02973.png', 'img/58306.png']
text (object, 7387 distinct): ['meanwhile at the isis strip club', 'when each letter is a mental disorder', 'we can kill as many as we want and your stupid government keeps bringing us in', 'sea monkeys', 'a head diaper is required when you have shit for brains', 'i only wear silk panties cotton ones remind me of slavery', 'this one time at camp we got so baked', 'how to get a black guy to see his baby', 'when your dishwasher is broken so you take it back to walmart to get a new one', 'mississippi wind chime']
'''

def load_df(dir_path: str) -> DataFrame:
    ret = []
    main_path = join(dir_path, IMAGE_FOLDER)
    for split in ["train", "dev", "test"]:
        split_path = join(main_path, f"{split}.jsonl")
        split_file = load_json_lines(split_path)
        for d in split_file:
            d.pop("id")
            ret.append(d)
    ret = DataFrame(ret)
    return ret


CONTEXT = "Facebook memes hateful content classification, based on image and text."
TARGET = CuratedTarget(raw_name="label", task_type=SupervisedTask.BINARY)
FEATURES = [CuratedFeature(raw_name="img", feat_type=FeatureType.IMAGE),
            CuratedFeature(raw_name="text", feat_type=FeatureType.TEXT)]
IMAGE_FOLDER = "data"
LOADING_FUNC = load_df
