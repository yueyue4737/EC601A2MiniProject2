# step1: overview tutorial
# Our goal is to train a classifier over the comment data that can predict whether a comment is spam or not spam.
# We have access to a large amount of *unlabeled data* in the form of YouTube comments with some metadata.
# In order to train a classifier, we need to label our data, but doing so by hand for real world applications can be prohibitively slow and expensive, often taking person-weeks or months.

# The tutorial is divided into four parts:
# 1. **Loading Data**: We load a [YouTube comments dataset](http://www.dt.fee.unicamp.br/~tiago//youtubespamcollection/), originally introduced in ["TubeSpam: Comment Spam Filtering on YouTube"](https://ieeexplore.ieee.org/document/7424299/), ICMLA'15 (T.C. Alberto, J.V. Lochter, J.V. Almeida).
#
# 2. **Writing Labeling Functions**: We write Python programs that take as input a data point and assign labels (or abstain) using heuristics, pattern matching, and third-party models.
#
# 3. **Combining Labeling Function Outputs with the Label Model**: We use the outputs of the labeling functions over the training set as input to the label model, which assigns probabilistic labels to the training set.
#
# 4. **Training a Classifier**: We train a classifier that can predict labels for *any* YouTube comment (not just the ones labeled by the labeling functions) using the probabilistic training labels from step 3.

import os

if os.path.basename(os.getcwd()) == "":
    os.chdir("")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

os.environ["PYTHONHASHSEED"] = "0"

import pandas as pd

DISPLAY_ALL_TEXT = False
pd.set_option("display.max_colwidth", 0 if DISPLAY_ALL_TEXT else 50)

from utils import load_spam_dataset

df_train, df_dev, df_valid, df_test = load_spam_dataset()

Y_dev = df_dev.label.values
Y_valid = df_valid.label.values
Y_test = df_test.label.values

df_dev.sample(5, random_state=3)

ABSTAIN = -1
HAM = 0
SPAM = 1

print(f"Dev SPAM frequency: {100 * (df_dev.label.values == SPAM).mean():.1f}%")

# ## 2. Writing Labeling Functions (LFs)

df_train[["author", "text", "video"]].sample(20, random_state=2)

from snorkel.labeling import labeling_function

@labeling_function()
def check(x):
    return SPAM if "check" in x.text.lower() else ABSTAIN

@labeling_function()
def check_out(x):
    return SPAM if "check out" in x.text.lower() else ABSTAIN

from snorkel.labeling import PandasLFApplier

lfs = [check_out, check]

applier = PandasLFApplier(lfs=lfs)
L_train = applier.apply(df=df_train)
L_dev = applier.apply(df=df_dev)

L_train

coverage_check_out, coverage_check = (L_train != ABSTAIN).mean(axis=0)
print(f"check_out coverage: {coverage_check_out * 100:.1f}%")
print(f"check coverage: {coverage_check * 100:.1f}%")

from snorkel.labeling import LFAnalysis

LFAnalysis(L=L_train, lfs=lfs).lf_summary()

LFAnalysis(L=L_dev, lfs=lfs).lf_summary(Y=Y_dev)

from snorkel.analysis import get_label_buckets

buckets = get_label_buckets(Y_dev, L_dev[:, 1])
df_dev.iloc[buckets[(HAM, SPAM)]]

df_train.iloc[L_train[:, 1] == SPAM].sample(10, random_state=1)

buckets = get_label_buckets(L_train[:, 0], L_train[:, 1])
df_train.iloc[buckets[(ABSTAIN, SPAM)]].sample(10, random_state=1)

import re

@labeling_function()
def regex_check_out(x):
    return SPAM if re.search(r"check.*out", x.text, flags=re.I) else ABSTAIN

lfs = [check_out, check, regex_check_out]

applier = PandasLFApplier(lfs=lfs)
L_train = applier.apply(df=df_train)
L_dev = applier.apply(df=df_dev)

LFAnalysis(L=L_train, lfs=lfs).lf_summary()

LFAnalysis(L_dev, lfs).lf_summary(Y=Y_dev)

buckets = get_label_buckets(L_dev[:, 1], L_dev[:, 2])
df_dev.iloc[buckets[(SPAM, ABSTAIN)]]

buckets = get_label_buckets(L_train[:, 1], L_train[:, 2])
df_train.iloc[buckets[(SPAM, ABSTAIN)]].sample(10, random_state=1)

from snorkel.preprocess import preprocessor
from textblob import TextBlob

@preprocessor(memoize=True)
def textblob_sentiment(x):
    scores = TextBlob(x.text)
    x.polarity = scores.sentiment.polarity
    x.subjectivity = scores.sentiment.subjectivity
    return x

import matplotlib.pyplot as plt

spam_polarities = [
    textblob_sentiment(x).polarity for _, x in df_dev.iterrows() if x.label == SPAM
]

ham_polarities = [
    textblob_sentiment(x).polarity for _, x in df_dev.iterrows() if x.label == HAM
]

plt.hist([spam_polarities, ham_polarities])
plt.title("TextBlob sentiment polarity scores")
plt.xlabel("Sentiment polarity score")
plt.ylabel("Count")
plt.legend(["Spam", "Ham"])
plt.show()

@labeling_function(pre=[textblob_sentiment])
def textblob_polarity(x):
    return HAM if x.polarity > 0.9 else ABSTAIN

spam_subjectivities = [
    textblob_sentiment(x).subjectivity for _, x in df_dev.iterrows() if x.label == SPAM
]

ham_subjectivities = [
    textblob_sentiment(x).subjectivity for _, x in df_dev.iterrows() if x.label == HAM
]

plt.hist([spam_subjectivities, ham_subjectivities])
plt.title("TextBlob sentiment subjectivity scores")
plt.xlabel("Sentiment subjectivity score")
plt.ylabel("Count")
plt.legend(["Spam", "Ham"])
plt.show()

plt.hist([spam_subjectivities, ham_subjectivities], bins=[0, 0.5, 1])
plt.title("TextBlob sentiment subjectivity scores")
plt.xlabel("Sentiment subjectivity score")
plt.ylabel("Count")
plt.legend(["Spam", "Ham"])
plt.show()

@labeling_function(pre=[textblob_sentiment])
def textblob_subjectivity(x):
    return HAM if x.subjectivity >= 0.5 else ABSTAIN

lfs = [textblob_polarity, textblob_subjectivity]

applier = PandasLFApplier(lfs)
L_train = applier.apply(df_train)
L_dev = applier.apply(df_dev)

LFAnalysis(L_train, lfs).lf_summary()

LFAnalysis(L_dev, lfs).lf_summary(Y=Y_dev)

import LabelingFunction


def keyword_lookup(x, keywords, label):
    if any(word in x.text.lower() for word in keywords):
        return label
    return ABSTAIN


def make_keyword_lf(keywords, label=SPAM):
    return LabelingFunction(
        name=f"keyword_{keywords[0]}",
        f=keyword_lookup,
        resources=dict(keywords=keywords, label=label),
    )

keyword_my = make_keyword_lf(keywords=["my"])

keyword_subscribe = make_keyword_lf(keywords=["subscribe"])

keyword_link = make_keyword_lf(keywords=["http"])

keyword_please = make_keyword_lf(keywords=["please", "plz"])

keyword_song = make_keyword_lf(keywords=["song"], label=HAM)

@labeling_function()
def short_comment(x):
    """Ham comments are often short, such as 'cool video!'"""
    return HAM if len(x.text.split()) < 5 else ABSTAIN

from snorkel.preprocess.nlp import SpacyPreprocessor

spacy = SpacyPreprocessor(text_field="text", doc_field="doc", memoize=True)

@labeling_function(pre=[spacy])
def has_person(x):
    """Ham comments mention specific people and are short."""
    if len(x.doc) < 20 and any([ent.label_ == "PERSON" for ent in x.doc.ents]):
        return HAM
    else:
        return ABSTAIN

from snorkel.labeling.lf.nlp import nlp_labeling_function

@nlp_labeling_function()
def has_person_nlp(x):
    """Ham comments mention specific people and are short."""
    if len(x.doc) < 20 and any([ent.label_ == "PERSON" for ent in x.doc.ents]):
        return HAM
    else:
        return ABSTAIN

lfs = [
    keyword_my,
    keyword_subscribe,
    keyword_link,
    keyword_please,
    keyword_song,
    regex_check_out,
    short_comment,
    has_person_nlp,
    textblob_polarity,
    textblob_subjectivity,
]

applier = PandasLFApplier(lfs=lfs)
L_train = applier.apply(df=df_train)
L_dev = applier.apply(df=df_dev)
L_valid = applier.apply(df=df_valid)

LFAnalysis(L=L_dev, lfs=lfs).lf_summary(Y=Y_dev)

def plot_label_frequency(L):
    plt.hist((L != ABSTAIN).sum(axis=1), density=True, bins=range(L.shape[1]))
    plt.xlabel("Number of labels")
    plt.ylabel("Fraction of dataset")
    plt.show()


plot_label_frequency(L_train)

from snorkel.labeling import MajorityLabelVoter

majority_model = MajorityLabelVoter()
preds_train = majority_model.predict(L=L_train)

preds_train

from snorkel.labeling import LabelModel

label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L_train=L_train, n_epochs=1000, lr=0.001, log_freq=100, seed=123)

majority_acc = majority_model.score(L=L_valid, Y=Y_valid)["accuracy"]
print(f"{'Majority Vote Accuracy:':<25} {majority_acc * 100:.1f}%")

label_model_acc = label_model.score(L=L_valid, Y=Y_valid)["accuracy"]
print(f"{'Label Model Accuracy:':<25} {label_model_acc * 100:.1f}%")

probs_dev = majority_model.predict_proba(L=L_dev)
preds_dev = probs_dev >= 0.5
buckets = get_label_buckets(Y_dev, preds_dev[:, 1])

df_fn_dev = df_dev[["text", "label"]].iloc[buckets[(SPAM, HAM)]]
df_fn_dev["probability"] = probs_dev[buckets[(SPAM, HAM)], 1]

df_fn_dev.sample(5, random_state=3)

def plot_probabilities_histogram(Y):
    plt.hist(Y, bins=10)
    plt.xlabel("Probability of SPAM")
    plt.ylabel("Number of data points")
    plt.show()


probs_train = label_model.predict_proba(L=L_train)
plot_probabilities_histogram(probs_train[:, SPAM])

from snorkel.labeling import filter_unlabeled_dataframe

df_train_filtered, probs_train_filtered = filter_unlabeled_dataframe(
    X=df_train, y=probs_train, L=L_train
)

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(ngram_range=(1, 2))
X_train = vectorizer.fit_transform(df_train_filtered.text.tolist())

X_dev = vectorizer.transform(df_dev.text.tolist())
X_valid = vectorizer.transform(df_valid.text.tolist())
X_test = vectorizer.transform(df_test.text.tolist())

import random

import numpy as np
import tensorflow as tf


seed = 1
np.random.seed(seed)
random.seed(seed)

session_conf = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
)

from tensorflow.keras import backend as K

tf.set_random_seed(seed)
sess = tf.compat.v1.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

from snorkel.analysis import metric_score
from snorkel.utils import preds_to_probs
from utils import get_keras_logreg, get_keras_early_stopping

keras_model = get_keras_logreg(input_dim=X_train.shape[1])

keras_model.fit(
    x=X_train,
    y=probs_train_filtered,
    validation_data=(X_valid, preds_to_probs(Y_valid, 2)),
    callbacks=[get_keras_early_stopping()],
    epochs=20,
    verbose=0,
)

preds_test = keras_model.predict(x=X_test).argmax(axis=1)
test_acc = metric_score(golds=Y_test, preds=preds_test, metric="accuracy")
print(f"Test Accuracy: {test_acc * 100:.1f}%")

keras_dev_model = get_keras_logreg(input_dim=X_train.shape[1], output_dim=1)

keras_dev_model.fit(
    x=X_dev,
    y=Y_dev,
    validation_data=(X_valid, Y_valid),
    callbacks=[get_keras_early_stopping()],
    epochs=20,
    verbose=0,
)

preds_test_dev = np.round(keras_dev_model.predict(x=X_test))
test_acc = metric_score(golds=Y_test, preds=preds_test_dev, metric="accuracy")
print(f"Test Accuracy: {test_acc * 100:.1f}%")

from snorkel.utils import probs_to_preds

preds_train_filtered = probs_to_preds(probs=probs_train_filtered)

from sklearn.linear_model import LogisticRegression

sklearn_model = LogisticRegression(C=0.001, solver="liblinear")
sklearn_model.fit(X=X_train, y=preds_train_filtered)

print(f"Test Accuracy: {sklearn_model.score(X=X_test, y=Y_test) * 100:.1f}%")

# step 2: augmentataion
# # 📈 Snorkel Intro Tutorial: Data Augmentation
# Data augmentation is a popular technique for increasing the size of labeled training sets by applying class-preserving transformations to create copies of labeled data points.
# In the image domain, it is a crucial factor in almost every state-of-the-art result today and is quickly gaining
# popularity in text-based applications.
# Snorkel models the data augmentation process by applying user-defined *transformation functions* (TFs) in sequence.
# You can learn more about data augmentation in
# [this blog post about our NeurIPS 2017 work on automatically learned data augmentation](https://snorkel.org/tanda/).

# The tutorial is divided into four parts:
# 1. **Loading Data**: We load a [YouTube comments dataset](http://www.dt.fee.unicamp.br/~tiago//youtubespamcollection/).
# 2. **Writing Transformation Functions**: We write Transformation Functions (TFs) that can be applied to training data points to generate new training data points.
# 3. **Applying Transformation Functions to Augment Our Dataset**: We apply a sequence of TFs to each training data point, using a random policy, to generate an augmented training set.
# 4. **Training a Model**: We use the augmented training set to train an LSTM model for classifying new comments as `SPAM` or `HAM`.

import os
import random

import numpy as np

if os.path.basename(os.getcwd()) == "":
    os.chdir("")

# Turn off TensorFlow logging messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# For reproducibility
seed = 0
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(0)
random.seed(0)

import pandas as pd

DISPLAY_ALL_TEXT = False

pd.set_option("display.max_colwidth", 0 if DISPLAY_ALL_TEXT else 50)

# This next cell makes sure a spaCy English model is downloaded.
# If this is your first time downloading this model, restart the kernel after executing the next cell.

# Download the spaCy english model
# ! python -m spacy download en_core_web_sm

# ## 1. Loading Data

# We load the Kaggle dataset and create Pandas DataFrame objects for each of the sets described above.
# The two main columns in the DataFrames are:
# * **`text`**: Raw text content of the comment
# * **`label`**: Whether the comment is `SPAM` (1) or `HAM` (0).

from utils import load_spam_dataset

df_train, _, df_valid, df_test = load_spam_dataset(load_train_labels=True)

# We pull out the label vectors for ease of use later
Y_valid = df_valid["label"].values
Y_train = df_train["label"].values
Y_test = df_test["label"].values

df_train.head()

# ## 2. Writing Transformation Functions (TFs)
# Transformation functions are functions that can be applied to a training data point to create another valid training data point of the same class.

from snorkel.preprocess.nlp import SpacyPreprocessor

spacy = SpacyPreprocessor(text_field="text", doc_field="doc", memoize=True)

import names
from snorkel.augmentation import transformation_function

# Pregenerate some random person names to replace existing ones with
# for the transformation strategies below
replacement_names = [names.get_full_name() for _ in range(50)]

# Replace a random named entity with a different entity of the same type.
@transformation_function(pre=[spacy])
def change_person(x):
    person_names = [ent.text for ent in x.doc.ents if ent.label_ == "PERSON"]
    # If there is at least one person name, replace a random one. Else return None.
    if person_names:
        name_to_replace = np.random.choice(person_names)
        replacement_name = np.random.choice(replacement_names)
        x.text = x.text.replace(name_to_replace, replacement_name)
        return x

# Swap two adjectives at random.
@transformation_function(pre=[spacy])
def swap_adjectives(x):
    adjective_idxs = [i for i, token in enumerate(x.doc) if token.pos_ == "ADJ"]
    # Check that there are at least two adjectives to swap.
    if len(adjective_idxs) >= 2:
        idx1, idx2 = sorted(np.random.choice(adjective_idxs, 2, replace=False))
        # Swap tokens in positions idx1 and idx2.
        x.text = " ".join(
            [
                x.doc[:idx1].text,
                x.doc[idx2].text,
                x.doc[1 + idx1 : idx2].text,
                x.doc[idx1].text,
                x.doc[1 + idx2 :].text,
            ]
        )
        return x

import nltk
from nltk.corpus import wordnet as wn

nltk.download("wordnet")

def get_synonym(word, pos=None):
    """Get synonym for word given its part-of-speech (pos)."""
    synsets = wn.synsets(word, pos=pos)
    # Return None if wordnet has no synsets (synonym sets) for this word and pos.
    if synsets:
        words = [lemma.name() for lemma in synsets[0].lemmas()]
        if words[0].lower() != word.lower():  # Skip if synonym is same as word.
            # Multi word synonyms in wordnet use '_' as a separator e.g. reckon_with. Replace it with space.
            return words[0].replace("_", " ")

def replace_token(spacy_doc, idx, replacement):
    """Replace token in position idx with replacement."""
    return " ".join([spacy_doc[:idx].text, replacement, spacy_doc[1 + idx :].text])

@transformation_function(pre=[spacy])
def replace_verb_with_synonym(x):
    # Get indices of verb tokens in sentence.
    verb_idxs = [i for i, token in enumerate(x.doc) if token.pos_ == "VERB"]
    if verb_idxs:
        # Pick random verb idx to replace.
        idx = np.random.choice(verb_idxs)
        synonym = get_synonym(x.doc[idx].text, pos="v")
        # If there's a valid verb synonym, replace it. Otherwise, return None.
        if synonym:
            x.text = replace_token(x.doc, idx, synonym)
            return x

@transformation_function(pre=[spacy])
def replace_noun_with_synonym(x):
    # Get indices of noun tokens in sentence.
    noun_idxs = [i for i, token in enumerate(x.doc) if token.pos_ == "NOUN"]
    if noun_idxs:
        # Pick random noun idx to replace.
        idx = np.random.choice(noun_idxs)
        synonym = get_synonym(x.doc[idx].text, pos="n")
        # If there's a valid noun synonym, replace it. Otherwise, return None.
        if synonym:
            x.text = replace_token(x.doc, idx, synonym)
            return x

@transformation_function(pre=[spacy])
def replace_adjective_with_synonym(x):
    # Get indices of adjective tokens in sentence.
    adjective_idxs = [i for i, token in enumerate(x.doc) if token.pos_ == "ADJ"]
    if adjective_idxs:
        # Pick random adjective idx to replace.
        idx = np.random.choice(adjective_idxs)
        synonym = get_synonym(x.doc[idx].text, pos="a")
        # If there's a valid adjective synonym, replace it. Otherwise, return None.
        if synonym:
            x.text = replace_token(x.doc, idx, synonym)
            return x

tfs = [
    change_person,
    swap_adjectives,
    replace_verb_with_synonym,
    replace_noun_with_synonym,
    replace_adjective_with_synonym,
]

from utils import preview_tfs

preview_tfs(df_train, tfs)

# %% [markdown]
from snorkel.augmentation import RandomPolicy

random_policy = RandomPolicy(
    len(tfs), sequence_length=2, n_per_original=2, keep_original=True
)

from snorkel.augmentation import MeanFieldPolicy

mean_field_policy = MeanFieldPolicy(
    len(tfs),
    sequence_length=2,
    n_per_original=2,
    keep_original=True,
    p=[0.05, 0.05, 0.3, 0.3, 0.3],
)

from snorkel.augmentation import PandasTFApplier

tf_applier = PandasTFApplier(tfs, mean_field_policy)
df_train_augmented = tf_applier.apply(df_train)
Y_train_augmented = df_train_augmented["label"].values

print(f"Original training set size: {len(df_train)}")
print(f"Augmented training set size: {len(df_train_augmented)}")

# ## 4. Training A Model
# Our final step is to use the augmented data to train a model. We train an LSTM (Long Short Term Memory) model, which is a very standard architecture for text processing tasks.
# The next cell makes Keras results reproducible. You can ignore it.

import tensorflow as tf

session_conf = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
)

tf.compat.v1.set_random_seed(0)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

# Now we'll train our LSTM on both the original and augmented datasets to compare performance.

from utils import featurize_df_tokens, get_keras_lstm, get_keras_early_stopping

X_train = featurize_df_tokens(df_train)
X_train_augmented = featurize_df_tokens(df_train_augmented)
X_valid = featurize_df_tokens(df_valid)
X_test = featurize_df_tokens(df_test)

def train_and_test(
    X_train,
    Y_train,
    X_valid=X_valid,
    Y_valid=Y_valid,
    X_test=X_test,
    Y_test=Y_test,
    num_buckets=30000,
):
    # Define a vanilla LSTM model with Keras
    lstm_model = get_keras_lstm(num_buckets)
    lstm_model.fit(
        X_train,
        Y_train,
        epochs=25,
        validation_data=(X_valid, Y_valid),
        callbacks=[get_keras_early_stopping(5)],
        verbose=0,
    )
    preds_test = lstm_model.predict(X_test)[:, 0] > 0.5
    return (preds_test == Y_test).mean()

acc_augmented = train_and_test(X_train_augmented, Y_train_augmented)
acc_original = train_and_test(X_train, Y_train)

print(f"Test Accuracy (original training data): {100 * acc_original:.1f}%")
print(f"Test Accuracy (augmented training data): {100 * acc_augmented:.1f}%")
# step3: slicing
# # ✂️ Snorkel Intro Tutorial: _Data Slicing_
# In real-world applications, some model outcomes are often more important than others — e.g. vulnerable cyclist detections in an autonomous driving task, or, in our running **spam** application, potentially malicious link redirects to external websites.
# Traditional machine learning systems optimize for overall quality, which may be too coarse-grained.
# Models that achieve high overall performance might produce unacceptable failure rates on critical slices of the data — data subsets that might correspond to vulnerable cyclist detection in an autonomous driving task, or in our running spam detection application, external links to potentially malicious websites.
#
# In this tutorial, we:
# 1. **Introduce _Slicing Functions (SFs)_** as a programming interface
# 1. **Monitor** application-critical data subsets
# 2. **Improve model performance** on slices

# %% [markdown] {"tags": ["md-exclude"]}
# First, we'll set up our notebook for reproducibility and proper logging.

import logging
import os
from snorkel.utils import set_seed

# For reproducibility
os.environ["PYTHONHASHSEED"] = "0"
set_seed(111)

# Make sure we're running from the spam/ directory
if os.path.basename(os.getcwd()) == "":
    os.chdir("")

# To visualize logs
logger = logging.getLogger()
logger.setLevel(logging.WARNING)

import pandas as pd

DISPLAY_ALL_TEXT = False

pd.set_option("display.max_colwidth", 0 if DISPLAY_ALL_TEXT else 50)

# SFs are intended to be used *after the training set has already been labeled* by LFs (or by hand) in the training data pipeline.

from utils import load_spam_dataset

df_train, df_valid, df_test = load_spam_dataset(load_train_labels=True, split_dev=False)

# %% [markdown]
# ## 1. Write slicing functions
# We leverage *slicing functions* (SFs), which output binary _masks_ indicating whether an data point is in the slice or not.
# Each slice represents some noisily-defined subset of the data (corresponding to an SF) that we'd like to programmatically monitor.

import re
from snorkel.slicing import slicing_function

@slicing_function()
def short_link(x):
    """Returns whether text matches common pattern for shortened ".ly" links."""
    return bool(re.search(r"\w+\.ly", x.text))

sfs = [short_link]

# ### Visualize slices

from snorkel.slicing import slice_dataframe

short_link_df = slice_dataframe(df_valid, short_link)

short_link_df[["text", "label"]]

# ## 2. Monitor slice performance with [`Scorer.score_slices`](https://snorkel.readthedocs.io/en/master/packages/_autosummary/analysis/snorkel.analysis.Scorer.html#snorkel.analysis.Scorer.score_slices)
# In this section, we'll demonstrate how we might monitor slice performance on the `short_link` slice — this approach is compatible with _any modeling framework_.

# ### Train a simple classifier
# First, we featurize the data — as you saw in the introductory Spam tutorial, we can extract simple bag-of-words features and store them as numpy arrays.

from sklearn.feature_extraction.text import CountVectorizer
from utils import df_to_features

vectorizer = CountVectorizer(ngram_range=(1, 1))
X_train, Y_train = df_to_features(vectorizer, df_train, "train")
X_valid, Y_valid = df_to_features(vectorizer, df_valid, "valid")
X_test, Y_test = df_to_features(vectorizer, df_test, "test")

# We define a `LogisticRegression` model from `sklearn` and show how we might visualize these slice-specific scores.

from sklearn.linear_model import LogisticRegression

sklearn_model = LogisticRegression(C=0.001, solver="liblinear")
sklearn_model.fit(X=X_train, y=Y_train)
print(f"Test set accuracy: {100 * sklearn_model.score(X_test, Y_test):.1f}%")

from snorkel.utils import preds_to_probs

preds_test = sklearn_model.predict(X_test)
probs_test = preds_to_probs(preds_test, 2)

from snorkel.slicing import PandasSFApplier

applier = PandasSFApplier(sfs)
S_test = applier.apply(df_test)

from snorkel.analysis import Scorer

scorer = Scorer(metrics=["accuracy", "f1"])

scorer.score_slices(
    S=S_test, golds=Y_test, preds=preds_test, probs=probs_test, as_dataframe=True
)

# ### Write additional slicing functions (SFs)
# Slices are dynamic — as monitoring needs grow or change with new data distributions or application needs, an ML pipeline might require dozens, or even hundreds, of slices.

from snorkel.slicing import SlicingFunction, slicing_function
from snorkel.preprocess import preprocessor

# Keyword-based SFs
def keyword_lookup(x, keywords):
    return any(word in x.text.lower() for word in keywords)

def make_keyword_sf(keywords):
    return SlicingFunction(
        name=f"keyword_{keywords[0]}",
        f=keyword_lookup,
        resources=dict(keywords=keywords),
    )

keyword_subscribe = make_keyword_sf(keywords=["subscribe"])
keyword_please = make_keyword_sf(keywords=["please", "plz"])

# Regex-based SF
@slicing_function()
def regex_check_out(x):
    return bool(re.search(r"check.*out", x.text, flags=re.I))

# Heuristic-based SF
@slicing_function()
def short_comment(x):
    """Ham comments are often short, such as 'cool video!'"""
    return len(x.text.split()) < 5

# Leverage preprocessor in SF
from textblob import TextBlob

@preprocessor(memoize=True)
def textblob_sentiment(x):
    scores = TextBlob(x.text)
    x.polarity = scores.sentiment.polarity
    return x

@slicing_function(pre=[textblob_sentiment])
def textblob_polarity(x):
    return x.polarity > 0.9

# Again, we'd like to visualize data points in a particular slice. This time, we'll inspect the `textblob_polarity` slice.
# Most data points with high-polarity sentiments are strong opinions about the video — hence, they are usually relevant to the video, and the corresponding labels are $0$.
# We might define a slice here for *product and marketing reasons*, it's important to make sure that we don't misclassify very positive comments from good users.

polarity_df = slice_dataframe(df_valid, textblob_polarity)

polarity_df[["text", "label"]].head()

extra_sfs = [
    keyword_subscribe,
    keyword_please,
    regex_check_out,
    short_comment,
    textblob_polarity,
]

sfs = [short_link] + extra_sfs
slice_names = [sf.name for sf in sfs]

applier = PandasSFApplier(sfs)
S_test = applier.apply(df_test)

scorer.score_slices(
    S=S_test, golds=Y_test, preds=preds_test, probs=probs_test, as_dataframe=True
)

# ## 3. Improve slice performance
# In the following section, we demonstrate a modeling approach that we call _Slice-based Learning,_ which improves performance by adding extra slice-specific representational capacity to whichever model we're using.

# ### Set up modeling pipeline with [`SlicingClassifier`](https://snorkel.readthedocs.io/en/master/packages/_autosummary/slicing/snorkel.slicing.SlicingClassifier.html)

# First, we initialize a dataloaders for each split.

from utils import create_dict_dataloader

BATCH_SIZE = 64

train_dl = create_dict_dataloader(
    X_train, Y_train, "train", batch_size=BATCH_SIZE, shuffle=True
)
valid_dl = create_dict_dataloader(
    X_valid, Y_valid, "valid", batch_size=BATCH_SIZE, shuffle=False
)
test_dl = create_dict_dataloader(
    X_test, Y_test, "test", batch_size=BATCH_SIZE, shuffle=True
)

# We'll now initialize a [`SlicingClassifier`](https://snorkel.readthedocs.io/en/master/packages/_autosummary/slicing/snorkel.slicing.SlicingClassifier.html):
# * `base_architecture`: We define a simple Multi-Layer Perceptron (MLP) in Pytorch to serve as the primary representation architecture. We note that the `BinarySlicingClassifier` is **agnostic to the base architecture** — you might leverage a Transformer model for text, or a ResNet for images.
# * `head_dim`: identifies the final output feature dimension of the `base_architecture`
# * `slice_names`: Specify the slices that we plan to train on with this classifier.

from snorkel.slicing import SlicingClassifier
from utils import get_pytorch_mlp

# Define model architecture
bow_dim = X_train.shape[1]
hidden_dim = bow_dim
mlp = get_pytorch_mlp(hidden_dim=hidden_dim, num_layers=2)

# Init slice model
slice_model = SlicingClassifier(
    base_architecture=mlp, head_dim=hidden_dim, slice_names=[sf.name for sf in sfs]
)

from snorkel.classification import Trainer

trainer = Trainer(lr=1e-4, n_epochs=2)
trainer.fit(slice_model, [train_dl, valid_dl])

applier = PandasSFApplier(sfs)
S_train = applier.apply(df_train)
S_valid = applier.apply(df_valid)

train_dl_slice = slice_model.make_slice_dataloader(
    train_dl.dataset, S_train, shuffle=True, batch_size=BATCH_SIZE
)
valid_dl_slice = slice_model.make_slice_dataloader(
    valid_dl.dataset, S_valid, shuffle=False, batch_size=BATCH_SIZE
)
test_dl_slice = slice_model.make_slice_dataloader(
    test_dl.dataset, S_test, shuffle=False, batch_size=BATCH_SIZE
)

# We train a single model initialized with all slice tasks.

from snorkel.classification import Trainer

trainer = Trainer(n_epochs=2, lr=1e-4, progress_bar=True)
trainer.fit(slice_model, [train_dl_slice, valid_dl_slice])

slice_model.score_slices([valid_dl_slice, test_dl_slice], as_dataframe=True)
